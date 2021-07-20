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

import torch

from flash.core.integrations.icevision.backbones import (
    get_backbones,
    icevision_model_adapter,
    load_icevision_ignore_image_size,
    load_icevision_with_image_size,
)
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _ICEVISION_AVAILABLE, _module_available, _TORCHVISION_AVAILABLE

if _ICEVISION_AVAILABLE:
    from icevision import models as icevision_models

OBJECT_DETECTION_HEADS = FlashRegistry("heads")

if _ICEVISION_AVAILABLE:
    if _TORCHVISION_AVAILABLE:
        for model_type in [icevision_models.torchvision.retinanet, icevision_models.torchvision.faster_rcnn]:
            OBJECT_DETECTION_HEADS(
                partial(load_icevision_ignore_image_size, icevision_model_adapter, model_type),
                model_type.__name__.split(".")[-1],
                backbones=get_backbones(model_type),
            )

    if _module_available("yolov5"):
        model_type = icevision_models.ultralytics.yolov5
        OBJECT_DETECTION_HEADS(
            partial(load_icevision_with_image_size, icevision_model_adapter, model_type),
            model_type.__name__.split(".")[-1],
            backbones=get_backbones(model_type),
        )

    if _module_available("mmdet"):
        for model_type in [
            icevision_models.mmdet.faster_rcnn,
            icevision_models.mmdet.retinanet,
            icevision_models.mmdet.fcos,
            icevision_models.mmdet.sparse_rcnn,
        ]:
            OBJECT_DETECTION_HEADS(
                partial(load_icevision_ignore_image_size, icevision_model_adapter, model_type),
                f"mmdet_{model_type.__name__.split('.')[-1]}",
                backbones=get_backbones(model_type),
            )

    if _module_available("effdet"):

        def _icevision_effdet_model_adapter(model_type):

            class IceVisionEffdetModelAdapter(icevision_model_adapter(model_type)):

                def validation_step(self, batch, batch_idx):
                    images = batch[0][0]
                    batch[0][1]["img_scale"] = torch.ones_like(images[:, 0, 0, 0]).unsqueeze(1)
                    batch[0][1]["img_size"] = (torch.ones_like(images[:, 0, 0, 0]) *
                                               images[0].shape[-1]).unsqueeze(1).repeat(1, 2)
                    return super().validation_step(batch, batch_idx)

            return IceVisionEffdetModelAdapter

        model_type = icevision_models.ross.efficientdet
        OBJECT_DETECTION_HEADS(
            partial(load_icevision_with_image_size, _icevision_effdet_model_adapter, model_type),
            model_type.__name__.split(".")[-1],
            backbones=get_backbones(model_type),
        )
