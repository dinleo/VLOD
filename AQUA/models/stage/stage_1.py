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

    def forward(self, batched_input):
        # Process images
        images = [i['image'] for i in batched_input]
        gt_instances = [i['instances']for i in batched_input]
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
                detr_output = self.detr_backbone(batched_input)
                aqua_input['multiscale_pred_logits'] = [detr_output['pred_logits']]
                aqua_input['multiscale_pred_boxes'] = [detr_output['pred_boxes']]
                aqua_input['multiscale_region_features'] = detr_output['hs']
                text_output = self.detr_backbone.bert_output(gt_captions)['bert_output']
            else:
                image_output =  self.image_backbone(batched_input)
                aqua_input['image_features'] = image_output['last_hidden_state']
                text_output = self.text_backbone(batched_input)

            text_embeds = text_output['last_hidden_state'].transpose(1, 2)
            class_embeds = self.post_logit(text_embeds, gt_captions) # [Batch, Class in prompt, 768]

        # Aqua
        aqua_output = self.aqua(aqua_input)
        kformer_output = aqua_output['kformer_output']['last_hidden_state'] # [Batch, Region Query, 768]
        gt_labels = aqua_output['gt_labels']

        # Make target
        targets = []
        for b in range(len(gt_labels)):
            labels = gt_labels[b]  # (Q,)
            class_embed = class_embeds[b]  # (C, 768)
            c2t = class_to_token_idx[b]  # dict: {class_id → token_idx}
            target_vecs = []

            for l in labels:
                cls = l.item()
                if cls == -1:
                    continue
                assert cls in c2t

                token_idx = c2t[cls]
                target_vecs.append(class_embed[token_idx])  # (768,)

            if target_vecs:
                target_tensor = torch.stack(target_vecs, dim=0)  # (GT_Instance, 768)
            else:
                target_tensor = torch.empty((0, class_embed.shape[-1]), device=class_embed.device)
            targets.append(target_tensor)

        output = {
            'kformer_output': kformer_output,
            'targets': targets,
        }
        return output

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

    def preprocess_image(self, batched_input):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_input]
        images = ImageList.from_tensors(images)

        return images



def build_stage1(args):
    model = safe_init(Stage1, args)
    model = load_model(model, args.ckpt_path)

    return model