import torch
import torch.nn as nn

from models.model_utils import load_model, safe_init, get_coco_cat_name
from models.post_process_logit import PostProcessLogit
from detectron2.structures import ImageList

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
            self.post_logit = PostProcessLogit(self.detr_backbone.tokenizer)
        else:
            self.detr_mode = False
            self.image_backbone = image_backbone
            self.text_backbone = text_backbone
            self.post_logit = PostProcessLogit(self.text_backbone.tokenizer)
        self.freeze_backbone()

    def forward(self, batched_inputs):
        # Process images
        images = [i['image'] for i in batched_inputs]
        gt_instances = [i['instances']for i in batched_inputs]
        aqua_input = {
            'images': images,
            'gt_instances': gt_instances,
            'image_features': None,
            'text_features': None,
        }

        # Make Captions for text backbone
        gt_captions = []
        class_to_token_idx = []
        for g in gt_instances:
            gt_class = list(set(g.gt_classes.detach().cpu().numpy()))
            cat_names = get_coco_cat_name(gt_class)
            caption = ". ".join(cat_names) + "."
            c2t = {}
            for i, c in enumerate(gt_class):
                c2t[c] = i
            gt_captions.append(caption)
            class_to_token_idx.append(c2t)

        # Backbone Inference
        with torch.no_grad():
            if self.detr_mode:
                self.detr_backbone.eval()
                detr_outputs = self.detr_backbone(batched_inputs)
                aqua_input['multiscale_pred_logits'] = [detr_outputs['pred_logits']]
                aqua_input['multiscale_pred_boxes'] = [detr_outputs['pred_boxes']]
                aqua_input['multiscale_region_features'] = detr_outputs['hs']
                text_output = self.detr_backbone.bert_output(gt_captions)['bert_output']
            else:
                image_output =  self.image_backbone(batched_inputs)
                aqua_input['image_features'] = image_output['last_hidden_state']
                text_output = self.text_backbone(batched_inputs)

        text_embeds = text_output['last_hidden_state'].transpose(1, 2)
        class_embeds = self.post_logit(text_embeds, gt_captions) # [Batch, Class in prompt, 768]
        # Aqua
        aqua_output = self.aqua(aqua_input)

        # Make target

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