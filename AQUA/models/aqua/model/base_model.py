"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import contextlib

import time
import datetime
import copy
import json
import logging
import sys

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from models.aqua.util import dist_utils as dist_utils
from models.aqua.util.logger import MetricLogger
from transformers import BertConfig


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.missing_keys = missing_keys
        self.unexpected_keys = unexpected_keys

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def get_optimizer_params(self, weight_decay):
        params_with_decay = []
        params_without_decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or name.endswith(".bias"):
                params_without_decay.append(param)
            else:
                params_with_decay.append(param)
        return [{"params": params_with_decay, "weight_decay": weight_decay},
            {"params": params_without_decay, "weight_decay": 0.0}, ]


def compute_sim_matrix(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = model.tokenizer(text, padding="max_length", truncation=True, max_length=35,
            return_tensors="pt", ).to(model.device)
        text_feat = model.forward_text(text_input)
        text_embed = F.normalize(model.text_proj(text_feat))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    vit_feats = []
    image_embeds = []
    for samples in data_loader:
        image = samples["image"]

        image = image.to(model.device)
        image_feat, vit_feat = model.forward_image(image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)

        vit_feats.append(vit_feat.cpu())
        image_embeds.append(image_embed)

    vit_feats = torch.cat(vit_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = []
    for image_embed in image_embeds:
        sim_q2t = image_embed @ text_embeds.t()
        sim_i2t, _ = sim_q2t.max(0)
        sims_matrix.append(sim_i2t)
    sims_matrix = torch.stack(sims_matrix, dim=0)

    score_matrix_i2t = torch.full((len(data_loader.dataset.image), len(texts)), -100.0).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device)
        score = model.compute_itm(image_inputs=image_inputs, text_ids=text_ids[topk_idx],
            text_atts=text_atts[topk_idx], ).float()
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
        score = model.compute_itm(image_inputs=image_inputs, text_ids=text_ids[start + i].repeat(k_test, 1),
            text_atts=text_atts[start + i].repeat(k_test, 1), ).float()
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


### A wrapper to view attributes in the config
class BertConfigW(BertConfig):
    r"""
    This is the configuration class to store the configuration of a [`BertModel`] or a [`TFBertModel`]. It is used to
    instantiate a BERT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the BERT
    [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        q_dim (`int`, *optional*, defaults to 768):
            Dimensionality of the query.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.

    Examples:

    ```python
    >>> from transformers import BertConfig, BertModel

    >>> # Initializing a BERT google-bert/bert-base-uncased style configuration
    >>> configuration = BertConfig()

    >>> # Initializing a model (with random weights) from the google-bert/bert-base-uncased style configuration
    >>> model = BertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "bert"

    def __init__(self, vocab_size=30522, region_size=1408, q_size=768, kv_size=768, num_hidden_layers=12,
            num_attention_heads=12, intermediate_size=3072, hidden_act="gelu", hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02,
            layer_norm_eps=1e-12, pad_token_id=0, position_embedding_type="absolute", use_cache=True,
            classifier_dropout=None, add_cross_attention=True, cross_attention_freq=2, **kwargs, ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.region_size = region_size
        self.q_size = q_size
        self.kv_size = kv_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.add_cross_attention = add_cross_attention
        self.cross_attention_freq = cross_attention_freq


# Param that Newly added or Initialized
missing_keys = [# Layer 0
    "Kformer.encoder.layer.0.crossattention.self.query.weight",
    "Kformer.encoder.layer.0.crossattention.self.query.bias",
    "Kformer.encoder.layer.0.crossattention.self.value.weight",
    "Kformer.encoder.layer.0.crossattention.self.value.bias",

    # Layer 2
    "Kformer.encoder.layer.2.crossattention.self.query.weight",
    "Kformer.encoder.layer.2.crossattention.self.query.bias",
    "Kformer.encoder.layer.2.crossattention.self.value.weight",
    "Kformer.encoder.layer.2.crossattention.self.value.bias",

    # Layer 4
    "Kformer.encoder.layer.4.crossattention.self.query.weight",
    "Kformer.encoder.layer.4.crossattention.self.query.bias",
    "Kformer.encoder.layer.4.crossattention.self.value.weight",
    "Kformer.encoder.layer.4.crossattention.self.value.bias",

    # Layer 6
    "Kformer.encoder.layer.6.crossattention.self.query.weight",
    "Kformer.encoder.layer.6.crossattention.self.query.bias",
    "Kformer.encoder.layer.6.crossattention.self.value.weight",
    "Kformer.encoder.layer.6.crossattention.self.value.bias",

    # Layer 8
    "Kformer.encoder.layer.8.crossattention.self.query.weight",
    "Kformer.encoder.layer.8.crossattention.self.query.bias",
    "Kformer.encoder.layer.8.crossattention.self.value.weight",
    "Kformer.encoder.layer.8.crossattention.self.value.bias",

    # Layer 10
    "Kformer.encoder.layer.10.crossattention.self.query.weight",
    "Kformer.encoder.layer.10.crossattention.self.query.bias",
    "Kformer.encoder.layer.10.crossattention.self.value.weight",
    "Kformer.encoder.layer.10.crossattention.self.value.bias",

    # region
    # 'region_feature.fc.weight', 'region_feature.fc.bias',
]

# Param that Don't Use from BLIP2
unexpected_keys = [
    'vision_proj.weight', 'vision_proj.bias',
    'text_proj.weight', 'text_proj.bias',
    'itm_head.weight', 'itm_head.bias',
    'Qformer.bert.embeddings.position_ids',
    'Qformer.bert.embeddings.word_embeddings.weight',
    'Qformer.bert.embeddings.position_embeddings.weight',
    'Qformer.bert.embeddings.LayerNorm.weight', 'Qformer.bert.embeddings.LayerNorm.bias',
    'Qformer.bert.encoder.layer.0.crossattention.self.value.weight',
    'Qformer.bert.encoder.layer.0.crossattention.self.value.bias',
    'Qformer.bert.encoder.layer.0.intermediate_query.dense.weight',
    'Qformer.bert.encoder.layer.0.intermediate_query.dense.bias',
    'Qformer.bert.encoder.layer.0.output_query.dense.weight',
    'Qformer.bert.encoder.layer.0.output_query.dense.bias',
    'Qformer.bert.encoder.layer.0.output_query.LayerNorm.weight',
    'Qformer.bert.encoder.layer.0.output_query.LayerNorm.bias',
    'Qformer.bert.encoder.layer.1.intermediate_query.dense.weight',
    'Qformer.bert.encoder.layer.1.intermediate_query.dense.bias',
    'Qformer.bert.encoder.layer.1.output_query.dense.weight',
    'Qformer.bert.encoder.layer.1.output_query.dense.bias',
    'Qformer.bert.encoder.layer.1.output_query.LayerNorm.weight',
    'Qformer.bert.encoder.layer.1.output_query.LayerNorm.bias',
    'Qformer.bert.encoder.layer.2.crossattention.self.value.weight',
    'Qformer.bert.encoder.layer.2.crossattention.self.value.bias',
    'Qformer.bert.encoder.layer.2.intermediate_query.dense.weight',
    'Qformer.bert.encoder.layer.2.intermediate_query.dense.bias',
    'Qformer.bert.encoder.layer.2.output_query.dense.weight',
    'Qformer.bert.encoder.layer.2.output_query.dense.bias',
    'Qformer.bert.encoder.layer.2.output_query.LayerNorm.weight',
    'Qformer.bert.encoder.layer.2.output_query.LayerNorm.bias',
    'Qformer.bert.encoder.layer.3.intermediate_query.dense.weight',
    'Qformer.bert.encoder.layer.3.intermediate_query.dense.bias',
    'Qformer.bert.encoder.layer.3.output_query.dense.weight',
    'Qformer.bert.encoder.layer.3.output_query.dense.bias',
    'Qformer.bert.encoder.layer.3.output_query.LayerNorm.weight',
    'Qformer.bert.encoder.layer.3.output_query.LayerNorm.bias',
    'Qformer.bert.encoder.layer.4.crossattention.self.value.weight',
    'Qformer.bert.encoder.layer.4.crossattention.self.value.bias',
    'Qformer.bert.encoder.layer.4.intermediate_query.dense.weight',
    'Qformer.bert.encoder.layer.4.intermediate_query.dense.bias',
    'Qformer.bert.encoder.layer.4.output_query.dense.weight',
    'Qformer.bert.encoder.layer.4.output_query.dense.bias',
    'Qformer.bert.encoder.layer.4.output_query.LayerNorm.weight',
    'Qformer.bert.encoder.layer.4.output_query.LayerNorm.bias',
    'Qformer.bert.encoder.layer.5.intermediate_query.dense.weight',
    'Qformer.bert.encoder.layer.5.intermediate_query.dense.bias',
    'Qformer.bert.encoder.layer.5.output_query.dense.weight',
    'Qformer.bert.encoder.layer.5.output_query.dense.bias',
    'Qformer.bert.encoder.layer.5.output_query.LayerNorm.weight',
    'Qformer.bert.encoder.layer.5.output_query.LayerNorm.bias',
    'Qformer.bert.encoder.layer.6.crossattention.self.value.weight',
    'Qformer.bert.encoder.layer.6.crossattention.self.value.bias',
    'Qformer.bert.encoder.layer.6.intermediate_query.dense.weight',
    'Qformer.bert.encoder.layer.6.intermediate_query.dense.bias',
    'Qformer.bert.encoder.layer.6.output_query.dense.weight',
    'Qformer.bert.encoder.layer.6.output_query.dense.bias',
    'Qformer.bert.encoder.layer.6.output_query.LayerNorm.weight',
    'Qformer.bert.encoder.layer.6.output_query.LayerNorm.bias',
    'Qformer.bert.encoder.layer.7.intermediate_query.dense.weight',
    'Qformer.bert.encoder.layer.7.intermediate_query.dense.bias',
    'Qformer.bert.encoder.layer.7.output_query.dense.weight',
    'Qformer.bert.encoder.layer.7.output_query.dense.bias',
    'Qformer.bert.encoder.layer.7.output_query.LayerNorm.weight',
    'Qformer.bert.encoder.layer.7.output_query.LayerNorm.bias',
    'Qformer.bert.encoder.layer.8.crossattention.self.value.weight',
    'Qformer.bert.encoder.layer.8.crossattention.self.value.bias',
    'Qformer.bert.encoder.layer.8.intermediate_query.dense.weight',
    'Qformer.bert.encoder.layer.8.intermediate_query.dense.bias',
    'Qformer.bert.encoder.layer.8.output_query.dense.weight',
    'Qformer.bert.encoder.layer.8.output_query.dense.bias',
    'Qformer.bert.encoder.layer.8.output_query.LayerNorm.weight',
    'Qformer.bert.encoder.layer.8.output_query.LayerNorm.bias',
    'Qformer.bert.encoder.layer.9.intermediate_query.dense.weight',
    'Qformer.bert.encoder.layer.9.intermediate_query.dense.bias',
    'Qformer.bert.encoder.layer.9.output_query.dense.weight',
    'Qformer.bert.encoder.layer.9.output_query.dense.bias',
    'Qformer.bert.encoder.layer.9.output_query.LayerNorm.weight',
    'Qformer.bert.encoder.layer.9.output_query.LayerNorm.bias',
    'Qformer.bert.encoder.layer.10.crossattention.self.value.weight',
    'Qformer.bert.encoder.layer.10.crossattention.self.value.bias',
    'Qformer.bert.encoder.layer.10.intermediate_query.dense.weight',
    'Qformer.bert.encoder.layer.10.intermediate_query.dense.bias',
    'Qformer.bert.encoder.layer.10.output_query.dense.weight',
    'Qformer.bert.encoder.layer.10.output_query.dense.bias',
    'Qformer.bert.encoder.layer.10.output_query.LayerNorm.weight',
    'Qformer.bert.encoder.layer.10.output_query.LayerNorm.bias',
    'Qformer.bert.encoder.layer.11.intermediate_query.dense.weight',
    'Qformer.bert.encoder.layer.11.intermediate_query.dense.bias',
    'Qformer.bert.encoder.layer.11.output_query.dense.weight',
    'Qformer.bert.encoder.layer.11.output_query.dense.bias',
    'Qformer.bert.encoder.layer.11.output_query.LayerNorm.weight',
    'Qformer.bert.encoder.layer.11.output_query.LayerNorm.bias', 'Qformer.cls.predictions.bias',
    'Qformer.cls.predictions.transform.dense.weight',
    'Qformer.cls.predictions.transform.dense.bias',
    'Qformer.cls.predictions.transform.LayerNorm.weight',
    'Qformer.cls.predictions.transform.LayerNorm.bias',
    'Qformer.cls.predictions.decoder.weight', 'Qformer.cls.predictions.decoder.bias'
]
