import torch
import torch.nn as nn

class MultiModelWrapper(nn.Module):
    def __init__(self, aqua, backbone, gdino):
        super().__init__()
        self.aqua = aqua
        self.backbone = backbone
        self.gdino = gdino

    def forward(self, samples):
        """
        samples: dict or tensor
        여기서 backbone → aqua → gdino 등 조합 자유롭게 가능
        """

        # 예시 흐름
        if self.backbone is not None:
            backbone_features = self.backbone(samples["image"])
            samples["backbone_features"] = backbone_features

        if self.aqua is not None:
            aqua_features = self.aqua(samples)
            samples["aqua_features"] = aqua_features

        if self.gdino is not None:
            gdino_outputs = self.gdino(samples)
            return gdino_outputs

        return samples

    def state_dict(self, *args, **kwargs):
        """
        Wrapper 자신이 아니라 내부 모델들의 state_dict을 모은다.
        """
        state = {}
        if self.aqua is not None:
            state['aqua'] = self.aqua.state_dict(*args, **kwargs)
        if self.backbone is not None:
            state['backbone'] = self.backbone.state_dict(*args, **kwargs)
        if self.gdino is not None:
            state['gdino'] = self.gdino.state_dict(*args, **kwargs)
        return state

    def load_state_dict(self, state_dict, strict=True):
        """
        로드할 때도 내부 모델별로 맞게 로드
        """
        if 'aqua' in state_dict and self.aqua is not None:
            self.aqua.load_state_dict(state_dict['aqua'], strict=strict)
        if 'backbone' in state_dict and self.backbone is not None:
            self.backbone.load_state_dict(state_dict['backbone'], strict=strict)
        if 'gdino' in state_dict and self.gdino is not None:
            self.gdino.load_state_dict(state_dict['gdino'], strict=strict)
        return

    def to(self, *args, **kwargs):
        """
        디바이스 이동도 내부 모델까지 자동으로 넘겨준다.
        """
        if self.aqua is not None:
            self.aqua = self.aqua.to(*args, **kwargs)
        if self.backbone is not None:
            self.backbone = self.backbone.to(*args, **kwargs)
        if self.gdino is not None:
            self.gdino = self.gdino.to(*args, **kwargs)
        return self

def build_stage1(args):
    return MultiModelWrapper(args)