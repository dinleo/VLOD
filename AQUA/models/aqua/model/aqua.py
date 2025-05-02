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
            region_size=1408,
            q_size=768,
            kv_size=768,
            num_kv_token=32,
    ):
        super().__init__()

        # Project vision feature to hidden_size
        self.region_query_generator = region_query_generator
        self.region_size = region_size
        self.q_size = q_size
        self.kv_size = kv_size
        self.num_kv_token = num_kv_token

        self.Kformer, self.kv_tokens = self.init_kformer()
        state_dict = self.Kformer.state_dict()
        for name, param in self.Kformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.ln_vision = nn.LayerNorm(region_size)

    def init_kformer(self):
        encoder_config = BertConfigW()
        encoder_config.region_size = self.region_size
        encoder_config.q_size = self.q_size
        encoder_config.kv_size = self.kv_size
        encoder_config.num_kv_token = self.num_kv_token

        kformer = Kformer(encoder_config)
        kv_token = nn.Parameter(
            torch.zeros(1, self.num_kv_token, encoder_config.kv_size)
        )
        kv_token.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        kformer.missing_keys = self.missing_keys
        kformer.unexpected_keys = self.unexpected_keys
        return kformer, kv_token


    def load_from_blip2(self, ckpt_path):
        blip2_state_dict = torch.load(ckpt_path, map_location="cpu")["model"]
        new_state_dict = {}
        kformer_state_dict = {}
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
                elif '.crossattention.' in new_k:   # include only QK of cross-attention
                    if '.self.query.' in new_k:         # query -> key
                        new_k = new_k.replace('.self.query.', '.self.key.');flag=True
                    elif '.self.key.' in new_k:         # key -> region
                        new_k = new_k.replace('.self.key.', '.self.region.');flag=True
                    elif 'self.value.' in new_k:        # discard value
                        pass
                else:
                    pass
                if flag:
                    new_k = new_k.replace('Qformer.bert.encoder', 'Kformer.encoder')
            new_state_dict[new_k] = v
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        pattern = re.compile(r"Kformer\.encoder\.layer\.\d+\.crossattention\.self\.(query|value)\.")

        for m in missing_keys:
            assert m.startswith('region_query_generator') or pattern.match(m)

        return ''

    def freeze_backbone(self):
        for param in self.region_query_generator.parameters():
            param.requires_grad = False
        self.region_query_generator.eval()

    def forward(self, dict_inputs):
        with torch.no_grad():
            outputs = self.region_query_generator(dict_inputs)
            nms_boxes = outputs['nms_boxes']
            nms_prob = outputs['nms_prob']
        visualize(nms_prob[0], nms_boxes[0], 'object .', dict_inputs['images'][0], threshold=0, is_cxcy=False, is_logit=False)
        # backbone_features = samples["backbone_features"]
        # multiscale_region_query = self.region_query_generator(backbone_features)
        # multiscale_region_query = self.ln_vision(multiscale_region_query)
        # kv_tokens = self.kv_tokens.expand(multiscale_region_query[0].shape[0], -1, -1)
        #
        # outputs = self.Kformer(
        #     multiscale_region_query=multiscale_region_query,
        #     kv_tokens=kv_tokens
        # )

        return outputs


def build_aqua(args):
    model = safe_init(AQuA, args)
    model = load_model(model, args.ckpt_path)
    if args.blip_ckpt_path:
        model.load_from_blip2(args.blip_ckpt_path)
    return model