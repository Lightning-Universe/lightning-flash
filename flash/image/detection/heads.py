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
from inspect import getmembers

import torch
from torch import nn

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _ICEVISION_AVAILABLE, _module_available, _TORCHVISION_AVAILABLE

if _ICEVISION_AVAILABLE:
    from icevision import models as icevision_models
    from icevision.backbones import BackboneConfig

OBJECT_DETECTION_HEADS = FlashRegistry("heads")

if _ICEVISION_AVAILABLE:

    def _icevision_model_adapter(model_type):

        class IceVisionModelAdapter(model_type.lightning.ModelAdapter):

            def log(self, name, value, **kwargs):
                if "prog_bar" not in kwargs:
                    kwargs["prog_bar"] = True
                return super().log(name, value, **kwargs)

        return IceVisionModelAdapter

    def _load_icevision(adapter, model_type, backbone, num_classes, **kwargs):
        model = model_type.model(backbone=backbone, num_classes=num_classes, **kwargs)

        backbone = nn.Module()
        params = model.param_groups()[0]
        for i, param in enumerate(params):
            backbone.register_parameter(f"backbone_{i}", param)

        return model_type, model, adapter(model_type), backbone

    def _load_icevision_ignore_image_size(adapter, model_type, backbone, num_classes, image_size=None, **kwargs):
        return _load_icevision(adapter, model_type, backbone, num_classes, **kwargs)

    def _load_icevision_with_image_size(adapter, model_type, backbone, num_classes, image_size=None, **kwargs):
        kwargs["img_size"] = image_size
        return _load_icevision(adapter, model_type, backbone, num_classes, **kwargs)

    def _get_backbones(model_type):
        _BACKBONES = FlashRegistry("backbones")

        for backbone_name, backbone_config in getmembers(model_type.backbones, lambda x: isinstance(x, BackboneConfig)):
            _BACKBONES(
                backbone_config,
                name=backbone_name,
            )
        return _BACKBONES

    if _TORCHVISION_AVAILABLE:
        for model_type in [icevision_models.torchvision.retinanet, icevision_models.torchvision.faster_rcnn]:
            OBJECT_DETECTION_HEADS(
                partial(_load_icevision_ignore_image_size, _icevision_model_adapter, model_type),
                model_type.__name__.split(".")[-1],
                backbones=_get_backbones(model_type),
            )

    if _module_available("yolov5"):
        model_type = icevision_models.ultralytics.yolov5
        OBJECT_DETECTION_HEADS(
            partial(_load_icevision_with_image_size, _icevision_model_adapter, model_type),
            model_type.__name__.split(".")[-1],
            backbones=_get_backbones(model_type),
        )

    if _module_available("mmdet"):
        for model_type in [
            icevision_models.mmdet.faster_rcnn,
            icevision_models.mmdet.retinanet,
            icevision_models.mmdet.fcos,
            icevision_models.mmdet.sparse_rcnn,
        ]:
            OBJECT_DETECTION_HEADS(
                partial(_load_icevision_ignore_image_size, _icevision_model_adapter, model_type),
                f"mmdet_{model_type.__name__.split('.')[-1]}",
                backbones=_get_backbones(model_type),
            )

    if _module_available("effdet"):

        def _icevision_effdet_model_adapter(model_type):

            class IceVisionEffdetModelAdapter(_icevision_model_adapter(model_type)):

                def validation_step(self, batch, batch_idx):
                    images = batch[0][0]
                    batch[0][1]["img_scale"] = torch.ones_like(images[:, 0, 0, 0]).unsqueeze(1)
                    batch[0][1]["img_size"] = (torch.ones_like(images[:, 0, 0, 0]) *
                                               images[0].shape[-1]).unsqueeze(1).repeat(1, 2)
                    return super().validation_step(batch, batch_idx)

            return IceVisionEffdetModelAdapter

        model_type = icevision_models.ross.efficientdet
        OBJECT_DETECTION_HEADS(
            partial(_load_icevision_with_image_size, _icevision_effdet_model_adapter, model_type),
            model_type.__name__.split(".")[-1],
            backbones=_get_backbones(model_type),
        )
