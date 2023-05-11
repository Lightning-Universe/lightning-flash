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
from inspect import getmembers

from torch import nn

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _ICEVISION_AVAILABLE

if _ICEVISION_AVAILABLE:
    from icevision.backbones import BackboneConfig


def load_icevision(model_type, backbone, num_classes, **kwargs):
    model = model_type.model(backbone=backbone, num_classes=num_classes, **kwargs)

    backbone = nn.Module()
    params = sum(model.param_groups()[:-1], [])
    for i, param in enumerate(params):
        backbone.register_parameter(f"backbone_{i}", param)

    # Param groups cause a pickle error so we remove them
    del model.param_groups
    if hasattr(model, "backbone") and hasattr(model.backbone, "param_groups"):
        del model.backbone.param_groups

    return model_type, model, model_type.lightning.ModelAdapter, backbone


def load_icevision_ignore_image_size(model_type, backbone, num_classes, image_size=None, **kwargs):
    return load_icevision(model_type, backbone, num_classes, **kwargs)


def load_icevision_with_image_size(model_type, backbone, num_classes, image_size=None, **kwargs):
    kwargs["img_size"] = image_size
    return load_icevision(model_type, backbone, num_classes, **kwargs)


def get_backbones(model_type):
    _BACKBONES = FlashRegistry("backbones")

    for backbone_name, backbone_config in getmembers(model_type.backbones, lambda x: isinstance(x, BackboneConfig)):
        # Only torchvision backbones with an FPN are supported
        if "torchvision" in model_type.__name__ and "fpn" not in backbone_name:
            continue

        _BACKBONES(
            backbone_config,
            name=backbone_name,
        )
    return _BACKBONES
