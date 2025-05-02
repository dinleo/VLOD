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
        detr_backbone=None,
        image_backbone=None,
        text_backbone=None,
        device="cuda"
    ):
        super().__init__()
        self.device = device

        self.aqua = aqua
        self.detr_backbone = detr_backbone
        if self.detr_backbone:
            self.detr_mode = True
            self.detr_backbone.mode = "stage1"
        else:
            self.detr_mode = False
            self.image_backbone = image_backbone
            self.text_backbone = text_backbone
        self.freeze_backbone()


    def forward(self, batched_inputs):
        # process images
        images = [i['image'] for i in batched_inputs]
        gt_instances = [i['instances']for i in batched_inputs]
        aqua_input = {
            'images': images,
            'gt_instances': gt_instances,
            'image_features': None,
            'text_features': None,
        }
        with torch.no_grad():
            if self.detr_mode:
                self.detr_backbone.eval()
                detr_outputs = self.detr_backbone(batched_inputs)
                aqua_input['multiscale_pred_logits'] = [detr_outputs['pred_logits']]
                aqua_input['multiscale_pred_boxes'] = [detr_outputs['pred_boxes']]
                aqua_input['multiscale_region_features'] = detr_outputs['hs']
            else:
                aqua_input['image_features'] = self.image_backbone(batched_inputs)
                aqua_input['text_features'] = self.text_backbone(batched_inputs)

        aqua_output = self.aqua(aqua_input)

        return aqua_output

    def freeze_backbone(self):
        if self.detr_mode:
            for param in self.detr_backbone.parameters():
                param.requires_grad = False
            self.detr_backbone.eval()
        else:
            for param in self.image_backbone.parameters():
                param.requires_grad = False
            self.image_backbone.eval()

            for param in self.text_backbone.parameters():
                param.requires_grad = False
            self.text_backbone.eval()

        self.aqua.freeze_backbone()

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)

        return images



def build_stage1(args):
    model = safe_init(Stage1, args)
    model = load_model(model, args.ckpt_path)

    return model