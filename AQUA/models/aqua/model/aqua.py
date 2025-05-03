"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import torch
import torch.nn as nn
import re

from models.aqua.model.base_model import BaseModel, BertConfigW
from models.aqua.model.kformer import Kformer
from models.model_utils import safe_init, load_model
from models.model_utils import visualize


class AQuA(BaseModel):
    def __init__(
            self,
            region_query_generator=None,
            region_size=256,
            q_size=768,
            kv_size=768,
            num_q_token=64,
            num_kv_token=32,
    ):
        super().__init__()

        # Project vision feature to hidden_size
        self.region_query_generator = region_query_generator
        self.region_size = region_size
        self.q_size = q_size
        self.kv_size = kv_size
        self.num_q_token = num_q_token
        self.num_kv_token = num_kv_token
        self.i = 0

        self.Kformer = self.init_kformer()
        self.kv_tokens = nn.Parameter(
            torch.zeros(1, self.num_kv_token, self.kv_size)
        )
        self.kv_tokens.data.normal_(mean=0.0, std=0.02)
        state_dict = self.Kformer.state_dict()
        for name, param in self.Kformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.region_projection = nn.Linear(region_size, q_size)
        # self.region_ln = nn.LayerNorm(region_size)

    def init_kformer(self):
        encoder_config = BertConfigW()
        encoder_config.region_size = self.region_size
        encoder_config.q_size = self.q_size
        encoder_config.kv_size = self.kv_size
        encoder_config.num_kv_token = self.num_kv_token

        kformer = Kformer(encoder_config)
        return kformer

    def load_from_blip2(self, ckpt_path):
        blip2_state_dict = torch.load(ckpt_path, map_location="cpu")["model"]
        new_state_dict = {}
        for k, v in blip2_state_dict.items():
            new_k = k
            if new_k == 'query_tokens':
                new_k = 'kv_tokens'
            elif new_k.startswith('Qformer.bert.encoder'):
                flag=False
                if '.intermediate.' in new_k:
                    new_k = new_k.replace('.intermediate.', '.activation.');flag=True
                elif '.output.' in new_k:
                    new_k = new_k.replace('.output.', '.drn.');flag=True
                elif '.attention.' in new_k:        # include all QKV of self-attention
                    new_k = new_k.replace('Qformer.bert.encoder', 'Kformer.encoder');flag=True
                elif '.crossattention.' in new_k:   # include only blip's Q to ours K
                    if '.self.query.' in new_k:
                        new_k = new_k.replace('.self.query.', '.self.key.');flag=True
                    # elif '.self.key.' in new_k:         # key -> region
                    #     new_k = new_k.replace('.self.key.', '.self.region.');flag=True
                    elif 'self.value.' in new_k or 'self.key' in new_k:        # discard value
                        pass
                else:
                    pass
                if flag:
                    new_k = new_k.replace('Qformer.bert.encoder', 'Kformer.encoder')
            new_state_dict[new_k] = v
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        pattern = re.compile(r"Kformer\.encoder\.layer\.\d+\.crossattention\.self\.(query|value|region)\.")

        for m in missing_keys:
            assert m.startswith('region') or pattern.match(m) or m.startswith('layerNorm'), m

        return ''

    def freeze_backbone(self):
        self.region_query_generator.eval()
        self.freeze_key()

    def freeze_key(self):
        for name, param in self.Kformer.named_parameters():
            pattern = re.compile(r"encoder\.layer\.\d+\.crossattention\.self\.key\.")
            if pattern.match(name):
                print(name)
                param.requires_grad=False
        self.kv_tokens.requires_grad = False

    def forward(self, dict_input):
        # visualize(dict_input['multiscale_pred_logits'][0][0], dict_input['multiscale_pred_boxes'][0][0], 'object .', dict_input['images'][0],
        #           threshold=0.01, is_cxcy=True, is_logit=True, save_name='raw')

        with torch.no_grad():
            nms_output = self.region_query_generator(dict_input)
            if nms_output is None:
                return None
            nms_prob = nms_output['nms_prob']
            nms_boxes = nms_output['nms_boxes']
            nms_index = nms_output['nms_index']
            gt_labels = nms_output['gt_labels']

        # visualize(nms_prob[0], nms_boxes[0], 'object .', dict_input['images'][0],
        #           threshold=0.01, is_cxcy=False, is_logit=False, save_name='nms')

        # Make Query (B, Q=64, D)
        multiscale_region_features = dict_input['multiscale_region_features']
        multiscale_region_query = []
        for feat in multiscale_region_features:
            # feat: (B, DinoQ=900, D)
            per_batch_queries = []
            for b in range(len(feat)):
                idx = nms_index[b]  # K=selected by nms including GT
                nms_region_query = feat[b][idx] # (K, D)
                region_query_b = self.pad_query(nms_region_query)  # pad or crop to (Q, D)
                per_batch_queries.append(region_query_b) # (Q, D)
            multiscale_region_query.append(torch.stack(per_batch_queries, dim=0)) # (B, num_q_token, D)

        # Make Query mask
        q_mask = []
        for labels in gt_labels:  # labels: Tensor of shape (K_b,)
            num_valid = min(len(labels), self.num_q_token)
            mask = torch.zeros(self.num_q_token, device=labels.device)
            mask[:num_valid] = 1.0
            q_mask.append(mask)
        q_mask = torch.stack(q_mask, dim=0)  # shape: (B, num_q_token)

        q_tokens = self.region_projection(multiscale_region_query[-1])
        kv_tokens = self.kv_tokens.expand(multiscale_region_query[-1].shape[0], -1, -1)

        kformer_output = self.Kformer(
            q_tokens=q_tokens,
            kv_tokens=kv_tokens,
            q_mask=q_mask,
            multiscale_region_query=multiscale_region_query,
        )
        output = {
            'kformer_output':kformer_output,
            'nms_index':nms_index,
            'gt_labels':gt_labels,
        }
        # if self.i == 98:
        #     self.region_query_generator.print_gt_matching()
        #     exit()
        # self.i +=1

        return output

    def pad_query(self, tensor):
        target_len = self.num_q_token
        Q, D = tensor.shape
        if Q == target_len:
            return tensor
        elif Q > target_len:
            return tensor[:target_len]
        else:
            pad_shape = list(tensor.shape)
            pad_shape[0] = target_len - Q
            pad_tensor = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, pad_tensor], dim=0)


def build_aqua(args):
    model = safe_init(AQuA, args)
    model = load_model(model, args.ckpt_path)
    if args.blip_ckpt_path:
        model.load_from_blip2(args.blip_ckpt_path)
    return model