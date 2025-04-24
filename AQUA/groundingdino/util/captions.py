import torch

class PostProcessCoco(torch.nn.Module):
    """ This module converts the model's output's (256 token logit) into (91 class sigmoid) for COCO evaluation"""

    def __init__(self, tokenlizer=None) -> None:
        super().__init__()
        self.tokenlizer = tokenlizer
        self.num_coco_cls = 91
        self.max_text_len = 256
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
        logit = logit.sigmoid()
        # (B, Query, Token) @ (B, Class, Token).T -> (B, Q, C)
        class_logit = logit @ pos_maps.transpose(1, 2)
        outputs["pred_logits"] = class_logit
        return outputs


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
    "person": 1,
    "bicycle": 2,
    "car": 3,
    "motorcycle": 4,
    "airplane": 5,
    "bus": 6,
    "train": 7,
    "truck": 8,
    "boat": 9,
    "traffic light": 10,
    "fire hydrant": 11,
    "stop sign": 13,
    "parking meter": 14,
    "bench": 15,
    "bird": 16,
    "cat": 17,
    "dog": 18,
    "horse": 19,
    "sheep": 20,
    "cow": 21,
    "elephant": 22,
    "bear": 23,
    "zebra": 24,
    "giraffe": 25,
    "backpack": 27,
    "umbrella": 28,
    "handbag": 31,
    "tie": 32,
    "suitcase": 33,
    "frisbee": 34,
    "skis": 35,
    "snowboard": 36,
    "sports ball": 37,
    "kite": 38,
    "baseball bat": 39,
    "baseball glove": 40,
    "skateboard": 41,
    "surfboard": 42,
    "tennis racket": 43,
    "bottle": 44,
    "wine glass": 46,
    "cup": 47,
    "fork": 48,
    "knife": 49,
    "spoon": 50,
    "bowl": 51,
    "banana": 52,
    "apple": 53,
    "sandwich": 54,
    "orange": 55,
    "broccoli": 56,
    "carrot": 57,
    "hot dog": 58,
    "pizza": 59,
    "donut": 60,
    "cake": 61,
    "chair": 62,
    "couch": 63,
    "potted plant": 64,
    "bed": 65,
    "dining table": 67,
    "toilet": 70,
    "tv": 72,
    "laptop": 73,
    "mouse": 74,
    "remote": 75,
    "keyboard": 76,
    "cell phone": 77,
    "microwave": 78,
    "oven": 79,
    "toaster": 80,
    "sink": 81,
    "refrigerator": 82,
    "book": 84,
    "clock": 85,
    "vase": 86,
    "scissors": 87,
    "teddy bear": 88,
    "hair drier": 89,
    "toothbrush": 90
}