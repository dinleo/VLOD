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

from .base_criterion import build_criterion
from .two_stage_criterion import build_criterion


def get_criterion(args):
    # we use register to maintain models from catdet6 on.
    from .registry import MODULE_BUILD_FUNCS

    assert args.criterion in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.criterion) # base_criterion => BaseCriterion
    criterion = build_func(args)
    return criterion