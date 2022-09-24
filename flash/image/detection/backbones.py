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
from typing import Optional

from lightning_utilities.core.imports import module_available

from flash.core.adapter import Adapter
from flash.core.integrations.icevision.adapter import IceVisionAdapter, SimpleCOCOMetric
from flash.core.integrations.icevision.backbones import (
    get_backbones,
    load_icevision_ignore_image_size,
    load_icevision_with_image_size,
)
from flash.core.model import Task
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _ICEVISION_AVAILABLE, _TORCHVISION_AVAILABLE
from flash.core.utilities.providers import _EFFDET, _ICEVISION, _MMDET, _TORCHVISION, _ULTRALYTICS

if _ICEVISION_AVAILABLE:
    from icevision import models as icevision_models
    from icevision.metrics import COCOMetricType
    from icevision.metrics import Metric as IceVisionMetric
else:

    class COCOMetricType:
        bbox = None


OBJECT_DETECTION_HEADS = FlashRegistry("heads")


class IceVisionObjectDetectionAdapter(IceVisionAdapter):
    @classmethod
    def from_task(
        cls,
        task: Task,
        num_classes: int,
        backbone: str = "resnet18_fpn",
        head: str = "retinanet",
        pretrained: bool = True,
        metrics: Optional["IceVisionMetric"] = SimpleCOCOMetric(COCOMetricType.bbox),
        image_size: Optional = None,
        **kwargs,
    ) -> Adapter:
        return super().from_task(
            task,
            num_classes=num_classes,
            backbone=backbone,
            head=head,
            pretrained=pretrained,
            metrics=metrics,
            image_size=image_size,
            **kwargs,
        )


if _ICEVISION_AVAILABLE:
    if _TORCHVISION_AVAILABLE:
        for model_type in [icevision_models.torchvision.retinanet, icevision_models.torchvision.faster_rcnn]:
            OBJECT_DETECTION_HEADS(
                partial(load_icevision_ignore_image_size, model_type),
                model_type.__name__.split(".")[-1],
                backbones=get_backbones(model_type),
                adapter=IceVisionObjectDetectionAdapter,
                providers=[_ICEVISION, _TORCHVISION],
            )

    if module_available("yolov5"):
        model_type = icevision_models.ultralytics.yolov5
        OBJECT_DETECTION_HEADS(
            partial(load_icevision_with_image_size, model_type),
            model_type.__name__.split(".")[-1],
            backbones=get_backbones(model_type),
            adapter=IceVisionObjectDetectionAdapter,
            providers=[_ICEVISION, _ULTRALYTICS],
        )

    if module_available("mmdet"):
        for model_type in [
            icevision_models.mmdet.faster_rcnn,
            icevision_models.mmdet.retinanet,
            icevision_models.mmdet.fcos,
            icevision_models.mmdet.sparse_rcnn,
        ]:
            OBJECT_DETECTION_HEADS(
                partial(load_icevision_ignore_image_size, model_type),
                f"mmdet_{model_type.__name__.split('.')[-1]}",
                backbones=get_backbones(model_type),
                adapter=IceVisionObjectDetectionAdapter,
                providers=[_ICEVISION, _MMDET],
            )

    if module_available("effdet"):

        model_type = icevision_models.ross.efficientdet
        OBJECT_DETECTION_HEADS(
            partial(load_icevision_with_image_size, model_type),
            model_type.__name__.split(".")[-1],
            backbones=get_backbones(model_type),
            adapter=IceVisionObjectDetectionAdapter,
            providers=[_ICEVISION, _EFFDET],
        )
