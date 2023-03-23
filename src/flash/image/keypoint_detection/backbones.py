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

from flash.core.adapter import Adapter
from flash.core.integrations.icevision.adapter import IceVisionAdapter
from flash.core.integrations.icevision.backbones import get_backbones, load_icevision_ignore_image_size
from flash.core.model import Task
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _ICEVISION_AVAILABLE, _TORCHVISION_AVAILABLE
from flash.core.utilities.providers import _ICEVISION, _TORCHVISION

if _ICEVISION_AVAILABLE:
    from icevision import models as icevision_models
    from icevision.metrics import Metric as IceVisionMetric

KEYPOINT_DETECTION_HEADS = FlashRegistry("heads")


class IceVisionKeypointDetectionAdapter(IceVisionAdapter):
    @classmethod
    def from_task(
        cls,
        task: Task,
        num_keypoints: int,
        num_classes: int = 2,
        backbone: str = "resnet18_fpn",
        head: str = "keypoint_rcnn",
        pretrained: bool = True,
        metrics: Optional["IceVisionMetric"] = None,
        image_size: Optional = None,
        **kwargs,
    ) -> Adapter:
        return super().from_task(
            task,
            num_keypoints=num_keypoints,
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
        model_type = icevision_models.torchvision.keypoint_rcnn
        KEYPOINT_DETECTION_HEADS(
            partial(load_icevision_ignore_image_size, model_type),
            model_type.__name__.split(".")[-1],
            backbones=get_backbones(model_type),
            adapter=IceVisionKeypointDetectionAdapter,
            providers=[_ICEVISION, _TORCHVISION],
        )
