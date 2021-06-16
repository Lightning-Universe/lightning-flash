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

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _TORCHVISION_AVAILABLE
from flash.image.backbones import catch_url_error

if _TORCHVISION_AVAILABLE:
    from torchvision.models import mobilenetv3, resnet

RESNET_MODELS = ["resnet50", "resnet101"]
MOBILENET_MODELS = ["mobilenet_v3_large"]

SEMANTIC_SEGMENTATION_BACKBONES = FlashRegistry("backbones")

if _TORCHVISION_AVAILABLE:

    def _load_resnet(model_name: str, pretrained: bool = True):
        backbone = resnet.__dict__[model_name](
            pretrained=pretrained,
            replace_stride_with_dilation=[False, True, True],
        )
        return backbone

    for model_name in RESNET_MODELS:
        SEMANTIC_SEGMENTATION_BACKBONES(
            fn=catch_url_error(partial(_load_resnet, model_name)),
            name=model_name,
            namespace="image/segmentation",
            package="torchvision",
        )

    def _load_mobilenetv3(model_name: str, pretrained: bool = True):
        backbone = mobilenetv3.__dict__[model_name](
            pretrained=pretrained,
            _dilated=True,
        )
        return backbone

    for model_name in MOBILENET_MODELS:
        SEMANTIC_SEGMENTATION_BACKBONES(
            fn=catch_url_error(partial(_load_mobilenetv3, model_name)),
            name=model_name,
            namespace="image/segmentation",
            package="torchvision",
        )
