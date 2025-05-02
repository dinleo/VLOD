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
            num_kv_token=32,
    ):
        super().__init__()

        # Project vision feature to hidden_size
        self.region_query_generator = region_query_generator
        self.region_size = region_size
        self.q_size = q_size
        self.kv_size = kv_size
        self.num_kv_token = num_kv_token
        self.i = 0

        self.Kformer, self.kv_tokens = self.init_kformer()
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
            assert m.startswith('region') or pattern.match(m) or m.startswith('layerNorm')

        return ''

    def freeze_backbone(self):
        for param in self.region_query_generator.parameters():
            param.requires_grad = False
        self.region_query_generator.eval()

    def forward(self, dict_inputs):
        # visualize(dict_inputs['multiscale_pred_logits'][0][0], dict_inputs['multiscale_pred_boxes'][0][0], 'object .', dict_inputs['images'][0],
        #           threshold=0.01, is_cxcy=True, is_logit=True, save_name='raw')

        with torch.no_grad():
            nms_outputs = self.region_query_generator(dict_inputs)
            if nms_outputs is None:
                return None
            nms_prob = nms_outputs['nms_prob']
            nms_boxes = nms_outputs['nms_boxes']
            nms_index = nms_outputs['nms_index']
            gt_labels = nms_outputs['gt_labels']

        # visualize(nms_prob[0], nms_boxes[0], 'object .', dict_inputs['images'][0],
        #           threshold=0.01, is_cxcy=False, is_logit=False, save_name='nms')

        multiscale_region_features = dict_inputs['multiscale_region_features']
        multiscale_region_query = []
        for feat in multiscale_region_features:
            # feat: (B, Q=900, D)
            B, Q, D = feat.shape
            idx = nms_index.unsqueeze(-1).expand(-1, -1, D)  # (B, K, D)
            region_query = torch.gather(feat, dim=1, index=idx)  # (B, K, D)
            multiscale_region_query.append(region_query)

        # multiscale_region_query = self.layerNorm_vision(multiscale_region_query)
        q_tokens = self.region_projection(multiscale_region_query[-1])
        kv_tokens = self.kv_tokens.expand(multiscale_region_query[-1].shape[0], -1, -1)

        kformer_outputs = self.Kformer(
            q_tokens=q_tokens,
            kv_tokens=kv_tokens,
            multiscale_region_query=multiscale_region_query,
        )
        outputs = {
            'kformer_outputs':kformer_outputs,
            'nms_index':nms_index,
            'gt_labels':gt_labels,
        }
        # if self.i == 98:
        #     self.region_query_generator.print_gt_matching()
        #     exit()
        # self.i +=1

        return outputs


def build_aqua(args):
    model = safe_init(AQuA, args)
    model = load_model(model, args.ckpt_path)
    if args.blip_ckpt_path:
        model.load_from_blip2(args.blip_ckpt_path)
    return model