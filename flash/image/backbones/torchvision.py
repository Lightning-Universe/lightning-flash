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
import torch.nn as nn

from functools import partial
from typing import Tuple

from flash.core.registry import FlashRegistry
from flash.image.backbones.utilities import catch_url_error
from flash.core.utilities.imports import _TORCHVISION_AVAILABLE
from flash.image.backbones.resnet import RESNET_MODELS

MOBILENET_MODELS = ["mobilenet_v2"]
VGG_MODELS = ["vgg11", "vgg13", "vgg16", "vgg19"]
DENSENET_MODELS = ["densenet121", "densenet169", "densenet161"]
TORCHVISION_MODELS = MOBILENET_MODELS + VGG_MODELS + RESNET_MODELS + DENSENET_MODELS


if _TORCHVISION_AVAILABLE:
    import torchvision
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

    def _fn_mobilenet_vgg(model_name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
        model: nn.Module = getattr(torchvision.models, model_name, None)(pretrained)
        backbone = model.features
        num_features = 512 if model_name in VGG_MODELS else model.classifier[-1].in_features
        return backbone, num_features

    def register_mobilenet_vgg_backbones(register: FlashRegistry):
        for model_name in MOBILENET_MODELS + VGG_MODELS:
            _type = "mobilenet" if model_name in MOBILENET_MODELS else "vgg"

            register(
                fn=catch_url_error(partial(_fn_mobilenet_vgg, model_name)),
                name=model_name,
                namespace="vision",
                package="torchvision",
                type=_type
            )

    def _fn_resnet_fpn(
        model_name: str,
        pretrained: bool = True,
        trainable_layers: bool = True,
        **kwargs,
    ) -> Tuple[nn.Module, int]:
        backbone = resnet_fpn_backbone(model_name, pretrained=pretrained, trainable_layers=trainable_layers, **kwargs)
        return backbone, 256

    def register_detection_backbones(register: FlashRegistry):
        for model_name in RESNET_MODELS:
            register(
                fn=catch_url_error(partial(_fn_resnet_fpn, model_name)),
                name=model_name,
                package="torchvision",
                type="resnet-fpn"
            )

    def _fn_densenet(model_name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
        model: nn.Module = getattr(torchvision.models, model_name, None)(pretrained)
        backbone = nn.Sequential(*model.features, nn.ReLU(inplace=True))
        num_features = model.classifier.in_features
        return backbone, num_features

    def register_densenet_backbones(register: FlashRegistry):
        for model_name in DENSENET_MODELS:
            register(
                fn=catch_url_error(partial(_fn_densenet, model_name)),
                name=model_name,
                namespace="vision",
                package="torchvision",
                type="densenet"
            )
