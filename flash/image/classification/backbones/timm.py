# Copyright The PyTorch Lightning team.
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
from functools import partial
from typing import Tuple

import torch.nn as nn

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _TIMM_AVAILABLE
from flash.core.utilities.providers import _TIMM
from flash.core.utilities.url_error import catch_url_error
from flash.image.classification.backbones.torchvision import TORCHVISION_MODELS

if _TIMM_AVAILABLE:
    import timm

    def _fn_timm(
        model_name: str,
        pretrained: bool = True,
        num_classes: int = 0,
        **kwargs,
    ) -> Tuple[nn.Module, int]:
        backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, **kwargs)
        num_features = backbone.num_features
        return backbone, num_features


def register_timm_backbones(register: FlashRegistry):
    if _TIMM_AVAILABLE:
        for model_name in timm.list_models():

            if model_name in TORCHVISION_MODELS:
                continue

            register(
                fn=catch_url_error(partial(_fn_timm, model_name)),
                name=model_name,
                namespace="vision",
                package="timm",
                providers=_TIMM,
            )
