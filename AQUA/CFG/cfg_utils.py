# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import pkg_resources
import datetime
from omegaconf import OmegaConf

from detectron2.config import LazyConfig
from detectron2.engine import default_setup

def try_get_key(cfg, *keys, default=None):
    """
    Try select keys from lazy cfg until the first key that exists. Otherwise return default.
    """
    for k in keys:
        none = object()
        p = OmegaConf.select(cfg, k, default=none)
        if p is not none:
            return p
    return default


def get_config(config_path):
    cfg_file = os.path.join("CFG", config_path)
    if not os.path.exists(cfg_file):
        raise RuntimeError("{} not available in configs!".format(config_path))
    cfg = LazyConfig.load(cfg_file)
    return cfg

def default_setup_detectron2(cfg, args):
    # detectron2's default_setup uses hardcoded keys within a specific configuration namespace
    # e.g., cfg.train.seed, cfg.train.output_dir, cfg.train.cudnn_benchmark, cfg.train.float32_precision, and args.eval_only

    cfg.train = cfg.runner
    args.eval_only = cfg.runner.eval_only
    date_str = datetime.datetime.now().strftime("%m%d_%H%M")
    cfg.runner.output_dir = os.path.join(cfg.runner.output_dir, cfg.runner.name, date_str)
    cfg.dataloader.evaluator.output_dir = cfg.runner.output_dir

    default_setup(cfg, args)