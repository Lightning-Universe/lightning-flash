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


def icevision_model_adapter(model_type):
    class IceVisionModelAdapter(model_type.lightning.ModelAdapter):
        def log(self, name, value, **kwargs):
            if "prog_bar" not in kwargs:
                kwargs["prog_bar"] = True
            return super().log(name.split("/")[-1], value, **kwargs)

    return IceVisionModelAdapter


def load_icevision(adapter, model_type, backbone, num_classes, **kwargs):
    model = model_type.model(backbone=backbone, num_classes=num_classes, **kwargs)

    backbone = nn.Module()
    params = model.param_groups()[0]
    for i, param in enumerate(params):
        backbone.register_parameter(f"backbone_{i}", param)

    return model_type, model, adapter(model_type), backbone


def load_icevision_ignore_image_size(adapter, model_type, backbone, num_classes, image_size=None, **kwargs):
    return load_icevision(adapter, model_type, backbone, num_classes, **kwargs)


def load_icevision_with_image_size(adapter, model_type, backbone, num_classes, image_size=None, **kwargs):
    kwargs["img_size"] = image_size
    return load_icevision(adapter, model_type, backbone, num_classes, **kwargs)


def get_backbones(model_type):
    _BACKBONES = FlashRegistry("backbones")

    for backbone_name, backbone_config in getmembers(model_type.backbones, lambda x: isinstance(x, BackboneConfig)):
        _BACKBONES(
            backbone_config,
            name=backbone_name,
        )
    return _BACKBONES
