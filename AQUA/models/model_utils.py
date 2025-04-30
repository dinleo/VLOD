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
from detectron2.config import instantiate
import inspect

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

        self._writer = wandb.init(
            entity=runner_cfg.wandb.entity,
            project=runner_cfg.wandb.project,
            name=runner_cfg.name,
            dir=runner_cfg.output_dir,
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


def load_model(build, ckpt):
    model = instantiate(build)
    if ckpt:
        checkpoint = torch.load(ckpt, map_location="cpu")
        _ = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)

    return model


def safe_init(cls, args: dict):
    sig = inspect.signature(cls.__init__)
    valid_keys = sig.parameters.keys() - {'self'}

    filtered_args = {k: v for k, v in args.items() if k in valid_keys}

    missing_keys = valid_keys - filtered_args.keys()
    if missing_keys:
        print(f"[Notice] Missing keys for {cls.__name__} (set default): {sorted(missing_keys)}")

    return cls(**filtered_args)


def check_frozen(model, max_depth=1):
    def collect_status(module):
        node = {}
        params = list(module.parameters())
        if params:
            node['Trainable Params'] = sum(p.requires_grad for p in params)
            node['Frozen Params'] = len(params) - node['Trainable Params']
        else:
            node['Trainable Params'] = 0
            node['Frozen Params'] = 0

        children = {}
        for child_name, child_module in module.named_children():
            children[child_name] = collect_status(child_module)

        if children:
            node['children'] = children

        return node

    tree = {}
    for name, module in model.named_children():
        tree[name] = collect_status(module)

    print_tree(tree, max_depth=max_depth)
    return tree

def print_tree(tree, depth=0, max_depth=1, prefix=""):
    if depth > max_depth:
        return
    if depth == 0:
        module_width = 40
        print(f"{'Module'.ljust(module_width)} | {'Trainable':>9} | {'Frozen':>6}")
        print("-" * (module_width + 20))

    module_width = 40
    total = len(tree)
    for idx, (module_name, node) in enumerate(tree.items()):
        trainable = node['Trainable Params']
        frozen = node['Frozen Params']

        if idx == total - 1:
            branch = "└── "
        else:
            branch = "├── "

        name_field = prefix + branch + module_name
        print(f"{name_field.ljust(module_width)} | {trainable:9} | {frozen:6}")

        if 'children' in node and depth < max_depth:
            if idx == total - 1:
                extension = "    "
            else:
                extension = "│   "
            print_tree(node['children'], depth=depth+1, max_depth=max_depth, prefix=prefix + extension)
