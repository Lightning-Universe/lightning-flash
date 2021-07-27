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

from flash.core.integrations.icevision.backbones import (
    get_backbones,
    icevision_model_adapter,
    load_icevision_ignore_image_size,
)
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _ICEVISION_AVAILABLE, _TORCHVISION_AVAILABLE
from flash.core.utilities.providers import _ICEVISION, _TORCHVISION

if _ICEVISION_AVAILABLE:
    from icevision import models as icevision_models

KEYPOINT_DETECTION_HEADS = FlashRegistry("heads")

if _ICEVISION_AVAILABLE:
    if _TORCHVISION_AVAILABLE:
        model_type = icevision_models.torchvision.keypoint_rcnn
        KEYPOINT_DETECTION_HEADS(
            partial(load_icevision_ignore_image_size, icevision_model_adapter, model_type),
            model_type.__name__.split(".")[-1],
            backbones=get_backbones(model_type),
            providers=[_ICEVISION, _TORCHVISION]
        )
