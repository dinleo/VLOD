# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# -*- coding: utf-8 -*-
# @Author: Yihao Chen
# @Date:   2021-08-16 16:03:17
# @Last Modified by:   Shilong Liu
# @Last Modified time: 2022-01-23 15:26
# modified from mmcv

import wandb
import torch
from PIL import Image
from collections import OrderedDict
from detectron2.utils.events import EventWriter, get_event_storage
from detectron2.config import LazyCall, instantiate
import inspect
from omegaconf import DictConfig

import cv2
import os
import supervision as sv
from torchvision.ops import box_convert
from dotenv import load_dotenv
load_dotenv()

class WandbWriter(EventWriter):
    """
    Write all scalars to a wandb file.
    """

    def __init__(self, runner_cfg, window_size: int = 20, **kwargs):
        """
        Args:
            log_dir (str): the directory to save the output events
            window_size (int): the scalars will be median-smoothed by this window size

            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self._window_size = window_size
        WDB = os.getenv('WANDB')
        if WDB:
            wandb.login(key=WDB)
        self._writer = wandb.init(
            entity=runner_cfg.wandb.entity,
            project=runner_cfg.wandb.project,
            name=runner_cfg.name,
            dir="wandb",
        )
        self._last_write = -1

    def write(self):
        storage = get_event_storage()
        new_last_write = self._last_write
        for k, (v, iter) in storage.latest_with_smoothing_hint(self._window_size).items():
            if iter > self._last_write:
                self._writer.log({k: v}, step=iter)
                new_last_write = max(new_last_write, iter)
        self._last_write = new_last_write

        # visualize training samples
        if len(storage._vis_data) >= 1:
            for img_name, img, step_num in storage._vis_data:
                log_img = Image.fromarray(img.transpose(1, 2, 0))  # convert to (h, w, 3) PIL.Image
                log_img = wandb.Image(log_img, caption=img_name)
                self._writer.log({img_name: [log_img]})
            # Storage stores all image data and rely on this writer to clear them.
            # As a result it assumes only one writer will use its image data.
            # An alternative design is to let storage store limited recent
            # data (e.g. only the most recent image) that all writers can access.
            # In that case a writer may not see all image data if its period is long.
            storage.clear_images()

    def close(self):
        if hasattr(self, "_writer"):
            self._writer.finish()


def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == "module.":
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict


def load_model(model, ckpt_path, strict=False):
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=strict)
        print(f"[Notice] [LOAD] {model.__class__.__name__} use checkpoint /{ckpt_path} ({len(missing)} missing, {len(unexpected)} unexpected)")
    return model


def safe_init(cls, args: dict):
    sig = inspect.signature(cls.__init__)
    valid_keys = sig.parameters.keys() - {'self'}

    filtered_args = {}
    for k, v in args.items():
        if k in valid_keys:
            if isinstance(v, (LazyCall, DictConfig)) and '_target_' in v:
                v = instantiate(v)
            filtered_args[k] = v

    missing_keys = valid_keys - filtered_args.keys()
    if missing_keys:
        print(f"[Notice] [INIT] {cls.__name__} Missing keys (set default): {sorted(missing_keys)}")

    return cls(**filtered_args)


def check_frozen(model, max_depth=1):
    def collect_status(module, name_path=""):
        own_params = list(module.named_parameters(recurse=False))

        node = {
            'Trainable Params': 0,
            'Frozen Params': 0,
            'Total Params': 0,
            'children': {}
        }

        for param_name, param in own_params:
            param_node = {
                'Trainable Params': int(param.requires_grad),
                'Frozen Params': int(not param.requires_grad),
                'Total Params': param.numel()
            }
            node['children'][param_name] = param_node
            node['Trainable Params'] += param_node['Trainable Params']
            node['Frozen Params'] += param_node['Frozen Params']
            node['Total Params'] += param_node['Total Params']

        for child_name, child_module in module.named_children():
            full_child_name = f"{name_path}.{child_name}" if name_path else child_name
            child_node = collect_status(child_module, full_child_name)
            node['children'][child_name] = child_node
            node['Trainable Params'] += child_node['Trainable Params']
            node['Frozen Params'] += child_node['Frozen Params']
            node['Total Params'] += child_node['Total Params']

        return node

    tree = {'__root__': collect_status(model)}
    print_tree(tree, max_depth=max_depth)
    return tree

def print_tree(tree, depth=0, max_depth=1, prefix=""):
    if depth > max_depth:
        return
    if depth == 0:
        module_width = 40
        print("-" * (module_width + 34))
        print(f"{'Module'.ljust(module_width)} | {'Trainable':>9} | {'Frozen':>6} |   Size  ")
        print("-" * (module_width + 34))

    module_width = 40
    total = len(tree)
    for idx, (module_name, node) in enumerate(tree.items()):
        trainable = node['Trainable Params']
        frozen = node['Frozen Params']
        total_params = format_params(node['Total Params'])

        if idx == total - 1:
            branch = "└── "
        else:
            branch = "├── "

        name_field = prefix + branch + module_name
        print(f"{name_field.ljust(module_width)} | {trainable:9} | {frozen:6} | {str(total_params).rjust(9)}")

        if 'children' in node and depth < max_depth:
            if idx == total - 1:
                extension = "    "
            else:
                extension = "│   "
            print_tree(node['children'], depth=depth+1, max_depth=max_depth, prefix=prefix + extension)

def format_params(num):
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)

def visualize(pred_logit, pred_boxes, caption, image, threshold=0.1, is_norm=True, is_logit=True, is_cxcy=True, save_name="result"):
    """
    Args:
        pred_logit: (N, C_prompt) logits before sigmoid
        pred_boxes: (N, 4) boxes in cxcywh format, normalized (0~1)
        caption: prompt string (e.g. "dog . cat . zebra .")
        image: torch Tensor [3, H, W], pixel image
        threshold: detection confidence threshold
    """

    # 1. Prepare
    cat_names = [c.strip() for c in caption.strip(" .").split(".") if c.strip()]
    image = image.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    h, w, _ = image.shape

    # 2. Process predictions
    if is_logit:
        pred_logit=pred_logit.sigmoid()
    if pred_logit.dim() == 1:
        pred_logit.unsqueeze_(1)
    pred_logit = pred_logit.detach().cpu()  # [N, C]
    pred_boxes = pred_boxes.detach().cpu()            # [N, 4], cxcywh (0~1)
    scores = pred_logit.max(dim=1).values    # [N]

    # 3. Filter by confidence
    mask = scores > threshold
    pred_logit = pred_logit[mask]        # [M, C]
    pred_boxes = pred_boxes[mask]        # [M, 4]
    scores = scores[mask]                # [M]

    # 4. Class index
    pred_class = pred_logit.argmax(dim=1).numpy()

    if pred_boxes.shape[0] == 0:
        print("No boxes passed threshold.")
        return image  # return unchanged

    # 5. Box conversion and scaling
    if is_cxcy:
        pred_boxes = box_convert(pred_boxes, in_fmt="cxcywh", out_fmt="xyxy")  # [M, 4]
    if is_norm:
        pred_boxes *= torch.tensor([w, h, w, h])  # scale to absolute coords
    pred_boxes = pred_boxes.numpy()

    # 6. Prepare visualization
    phrases = [cat_names[c] for c in pred_class]
    detections = sv.Detections(xyxy=pred_boxes)

    labels = [
        (f"GT-{coco_name_list[int(score-100)]}" if score >= 100 else f"{phrase} {score:.2f}")
        for phrase, score in zip(phrases, scores.numpy())
    ]

    box_annotator = sv.BoxAnnotator()
    annotated = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated = box_annotator.annotate(scene=annotated, detections=detections, labels=labels)
    num_gt = (scores >= 100).sum().item()
    print(f"predict {len(pred_class)} instances (GT: {num_gt})")
    cv2.imwrite(f"outputs/{save_name}.jpg", annotated)
    # cv2.imshow("Prediction", annotated)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return annotated


coco_name_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
def get_coco_cat_name(id_list):
    return [coco_name_list[i] for i in id_list]