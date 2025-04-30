import torch
import torch.nn as nn
from models.model_utils import load_model, safe_init


class Stage1(nn.Module):
    def __init__(self, aqua, groundingdino=None, text_backbone=None, image_backbone=None):
        super().__init__()
        self.aqua = aqua
        self.gdino = groundingdino
        self.text_backbone = text_backbone
        self.image_backbone = image_backbone

    def forward(self, samples):

        return self.gdino(samples)

    def freeze_aqua(self):
        for param in self.aqua.parameters():
            param.requires_grad = False

    # def state_dict(self, *args, **kwargs):
    #     """
    #     Wrapper 자신이 아니라 내부 모델들의 state_dict을 모은다.
    #     """
    #     state = {}
    #     if self.aqua is not None:
    #         state['aqua'] = self.aqua.state_dict(*args, **kwargs)
    #     if self.backbone is not None:
    #         state['backbone'] = self.backbone.state_dict(*args, **kwargs)
    #     if self.gdino is not None:
    #         state['gdino'] = self.gdino.state_dict(*args, **kwargs)
    #     return state

    # def load_state_dict(self, state_dict, strict=True):
    #     """
    #     로드할 때도 내부 모델별로 맞게 로드
    #     """
    #     if 'aqua' in state_dict and self.aqua is not None:
    #         self.aqua.load_state_dict(state_dict['aqua'], strict=strict)
    #     if 'backbone' in state_dict and self.backbone is not None:
    #         self.backbone.load_state_dict(state_dict['backbone'], strict=strict)
    #     if 'gdino' in state_dict and self.gdino is not None:
    #         self.gdino.load_state_dict(state_dict['gdino'], strict=strict)
    #     return

    # def to(self, *args, **kwargs):
    #     """
    #     디바이스 이동도 내부 모델까지 자동으로 넘겨준다.
    #     """
    #     if self.aqua is not None:
    #         self.aqua = self.aqua.to(*args, **kwargs)
    #     if self.backbone is not None:
    #         self.backbone = self.backbone.to(*args, **kwargs)
    #     if self.gdino is not None:
    #         self.gdino = self.gdino.to(*args, **kwargs)
    #     return self

def build_stage1(args):
    args.aqua = load_model(args.aqua.build, args.aqua.ckpt)
    args.groundingdino = load_model(args.groundingdino.build, args.groundingdino.ckpt)

    model = safe_init(Stage1, args)

    return model