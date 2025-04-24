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

from .criterion import SetCriterion
from .base_criterion import BaseCriterion
from .matcher import  HungarianMatcher, FocalLossCost, GIoUCost, L1Cost

def build_criterion(args):
    hm = HungarianMatcher(
        cost_class=FocalLossCost(
            alpha=0.25,
            gamma=2.0,
            weight=2.0,
        ),
        cost_bbox=L1Cost(weight=5.0),
        cost_giou=GIoUCost(weight=2.0),
    ),

    return BaseCriterion(num_classes=args.max_text_len, matcher=hm)
