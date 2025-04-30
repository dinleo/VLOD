import torch
import torch.nn as nn
from typing import List
from models.model_utils import load_model, safe_init
from detectron2.structures import ImageList
from models.groundingdino.util.misc import nested_tensor_from_tensor_list
from models.groundingdino.model.bertwarper import generate_masks_with_special_tokens_and_transfer_map
from models.groundingdino.util.box_ops import box_xyxy_to_cxcywh

class Stage1(nn.Module):
    def __init__(
        self,
        aqua,
        groundingdino,
        sub_sentence_present=True,
        max_text_len=256,
        pixel_mean: List[float] = [123.675, 116.280, 103.530],
        pixel_std: List[float] = [123.675, 116.280, 103.530],
        device="cuda"
    ):
        super().__init__()
        self.aqua = aqua
        self.text_backbone = groundingdino.bert
        self.image_backbone = groundingdino.backbone
        self.tokenizer = groundingdino.tokenizer
        self.normalizer = groundingdino.normalizer

        self.sub_sentence_present = sub_sentence_present
        self.max_text_len = max_text_len
        self.device = device
        self.special_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

        self.freeze_backbone()



    def forward(self, batched_inputs):
        # process images
        images = self.preprocess_image(batched_inputs)
        assert isinstance(images, ImageList)
        samples = nested_tensor_from_tensor_list(images)

        captions = [x["captions"] for x in batched_inputs]
        names_list = [x["captions"][:-1].split(".") for x in batched_inputs]

        # encoder texts
        tokenized = self.tokenizer(captions, padding="longest", return_tensors="pt").to(
            samples.device
        )
        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.special_tokens, self.tokenizer
        )

        # prepare targets
        targets = None
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances, cate_to_token_mask_list, names_list)

        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                                        :, : self.max_text_len, : self.max_text_len
                                        ]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

        # extract text embeddings
        if self.sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            # import ipdb; ipdb.set_trace()
            tokenized_for_encoder = tokenized

        bert_output = self.text_backbone(**tokenized_for_encoder)
        features, poss = self.image_backbone(samples)

        return

    def freeze_backbone(self):
        for param in self.text_backbone.parameters():
            param.requires_grad = False
        for param in self.image_backbone.parameters():
            param.requires_grad = False

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)

        return images

    def normalizer(self, x):
        pixel_mean = torch.Tensor(self.pixel_mean).to(x.device).view(3, 1, 1)
        pixel_std = torch.Tensor(self.pixel_std).to(x.device).view(3, 1, 1)
        return (x - pixel_mean) / pixel_std

    def prepare_targets(self, targets, cate_to_token_mask_list, names_list):
        new_targets = []
        for targets_per_image, cate_to_token_mask, names in \
            zip(targets, cate_to_token_mask_list, names_list):
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            # gt_class_names = targets_per_image.gt_class_names
            # gt_classes =  torch.as_tensor([names.index(name) for name in gt_class_names],
            #                                        dtype=torch.long, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
        return new_targets


def build_stage1(args):
    args.aqua = load_model(args.aqua.build, args.aqua.ckpt)
    args.groundingdino = load_model(args.groundingdino.build, args.groundingdino.ckpt)

    model = safe_init(Stage1, args)

    return model