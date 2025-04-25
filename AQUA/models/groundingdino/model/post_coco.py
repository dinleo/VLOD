import torch
from detectron2.structures import Boxes, Instances
from ..util.box_ops import box_cxcywh_to_xyxy

class PostProcessCoco(torch.nn.Module):
    """ This module converts the model's output's (256 token logit) into (91 class sigmoid) for COCO evaluation"""

    def __init__(self, tokenlizer=None, num_coco_cls=80, max_text_len=256, topk_num=300) -> None:
        super().__init__()
        self.tokenlizer = tokenlizer
        self.num_coco_cls = num_coco_cls
        self.max_text_len = max_text_len
        self.topk_num = topk_num
        self.pos_map_hub = {}

    def forward(self, inputs, outputs):
        """
        Change token logit to 91 class logit(sigmoid) for COCO evaluation.
        Args:
            inputs: list of data with key "captions" each like "person . dog . cat ."
            outputs: dict with key "pred_logits" of shape [B, Q, T]

        Returns:
            logits_cls: Tensor of shape [B, Q, C]
        """
        logit = outputs["pred_logits"]  # [B, Q, T]
        device = logit.device
        pos_maps = []

        for i in inputs:
            caption = i["captions"]
            pos_map = self.get_pos_map(caption)
            pos_maps.append(pos_map)


        pos_maps = torch.stack(pos_maps, dim=0).to(device)  # [B, C, T]
        logit[torch.isinf(logit)] = 0
        # (B, Query, Token) @ (B, Class, Token).T -> (B, Q, C)
        class_logit = logit @ pos_maps.transpose(1, 2)

        pos_map_mask = pos_maps.sum(dim=2)
        pos_map_mask = (pos_map_mask == 0).unsqueeze(1)
        class_logit = class_logit.masked_fill(pos_map_mask, float('-inf'))

        outputs["pred_logits"] = class_logit
        return outputs

    def select_topk(self, outputs, image_sizes):
        """
        Arguments:
            outputs have keys "pred_logits" and "pred_boxes"
                pred_logits (Tensor): tensor of shape (batch_size, num_queries, K).
                    The tensor predicts the classification probability for each query.
                pred_boxes (Tensor): tensors of shape (batch_size, num_queries, 4).
                    The tensor predicts 4-vector (x,y,w,h) box
                    regression values for every queryx
                image_sizes (List[torch.Size]): the input image sizes
        Returns:
            results (List[Instances]): a list of #images elements.
        """
        box_cls = outputs["pred_logits"]
        box_pred = outputs["pred_boxes"]
        assert len(box_cls) == len(image_sizes)
        results = []

        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1), self.topk_num, dim=1
        )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]

        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # For each box we assign the best class or the second best if the best on is `no_object`.
        # scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
                zip(scores, labels, boxes, image_sizes)
        ):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))

            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def get_pos_map(self, caption):
        """
        Create a positive map [C, T] for token-level alignment to category names.

        Args:
            tokenized: tokenizer output from `tokenizer(caption, return_offsets_mapping=True, ...)`
            caption (str): raw caption string (e.g., "bear . zebra . giraffe .")
        Returns:
            Tensor: positive_map [C, max_text_len]
        """
        if self.pos_map_hub.get(caption, None) is not None:
            return self.pos_map_hub[caption]

        tokenized = self.tokenlizer(caption, return_offsets_mapping=True, padding="longest", max_length=256)
        offsets = tokenized["offset_mapping"]
        if isinstance(offsets, torch.Tensor):
            offsets = offsets.tolist()

        cat_names = [c.strip() for c in caption.strip(" .").split(".") if c.strip()]
        positive_map = torch.zeros((len(cat_names), self.max_text_len), dtype=torch.float)

        curr_char_pos = 0
        for i, cat in enumerate(cat_names):
            try:
                cat_start = caption.index(cat, curr_char_pos)
                cat_end = cat_start + len(cat)
            except ValueError:
                continue

            for j, (start, end) in enumerate(offsets):
                if start is None or end is None:
                    continue
                if end <= cat_start:
                    continue
                if start >= cat_end:
                    break
                positive_map[i, j] = 1.0

            curr_char_pos = cat_end

        class_token_map = positive_map / (positive_map.sum(-1)[:, None] + 1e-6) # [Class# in prompt, T]
        coco_token_map = torch.zeros((self.num_coco_cls, self.max_text_len))  # [Class# in coco_val, T]
        for i, cat in enumerate(cat_names):
            coco_id = cat2id[cat]
            coco_token_map[coco_id] = class_token_map[i]

        self.pos_map_hub[caption] = coco_token_map
        return coco_token_map


cat2id = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "airplane": 4,
    "bus": 5,
    "train": 6,
    "truck": 7,
    "boat": 8,
    "traffic light": 9,
    "fire hydrant": 10,
    "stop sign": 11,
    "parking meter": 12,
    "bench": 13,
    "bird": 14,
    "cat": 15,
    "dog": 16,
    "horse": 17,
    "sheep": 18,
    "cow": 19,
    "elephant": 20,
    "bear": 21,
    "zebra": 22,
    "giraffe": 23,
    "backpack": 24,
    "umbrella": 25,
    "handbag": 26,
    "tie": 27,
    "suitcase": 28,
    "frisbee": 29,
    "skis": 30,
    "snowboard": 31,
    "sports ball": 32,
    "kite": 33,
    "baseball bat": 34,
    "baseball glove": 35,
    "skateboard": 36,
    "surfboard": 37,
    "tennis racket": 38,
    "bottle": 39,
    "wine glass": 40,
    "cup": 41,
    "fork": 42,
    "knife": 43,
    "spoon": 44,
    "bowl": 45,
    "banana": 46,
    "apple": 47,
    "sandwich": 48,
    "orange": 49,
    "broccoli": 50,
    "carrot": 51,
    "hot dog": 52,
    "pizza": 53,
    "donut": 54,
    "cake": 55,
    "chair": 56,
    "couch": 57,
    "potted plant": 58,
    "bed": 59,
    "dining table": 60,
    "toilet": 61,
    "tv": 62,
    "laptop": 63,
    "mouse": 64,
    "remote": 65,
    "keyboard": 66,
    "cell phone": 67,
    "microwave": 68,
    "oven": 69,
    "toaster": 70,
    "sink": 71,
    "refrigerator": 72,
    "book": 73,
    "clock": 74,
    "vase": 75,
    "scissors": 76,
    "teddy bear": 77,
    "hair drier": 78,
    "toothbrush": 79
}
