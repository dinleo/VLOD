import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from models.model_utils import load_model, safe_init
from detectron2.structures import ImageList
from models.groundingdino.util.misc import nested_tensor_from_tensor_list, NestedTensor
from models.groundingdino.model.bertwarper import generate_masks_with_special_tokens_and_transfer_map
from models.groundingdino.util.box_ops import box_xyxy_to_cxcywh

class Stage1(nn.Module):
    def __init__(
        self,
        aqua,
        backbone,
        device="cuda"
    ):
        super().__init__()
        self.device = device

        self.aqua = aqua
        self.backbone = backbone
        self.backbone.mode = "stage1"
        self.freeze_backbone()
        # TODO: image, text backbone seperated mode


    def forward(self, batched_inputs):
        # process images
        with torch.no_grad():
            self.backbone.eval()
            backbone_outputs = self.backbone(batched_inputs)
        image_sizes = [i['instances'].image_size for i in batched_inputs]

        if 'pred_logits' in backbone_outputs:
            # if backbone is groundingdino, Aqua don't need rpn
            aqua_input = {
                'pred_logits': backbone_outputs['pred_logits'],
                'pred_boxes': backbone_outputs['pred_boxes'],
                'backbone_features': backbone_outputs['hs'],
                'image_sizes' : image_sizes,
            }
        else:
            aqua_input = {
                'backbone_features': backbone_outputs,
                'image_sizes': image_sizes,
            }
        aqua_output = self.aqua(aqua_input)

        return

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self.aqua.freeze_backbone()

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)

        return images



def build_stage1(args):
    model = safe_init(Stage1, args)
    model = load_model(model, args.ckpt_path)

    return model