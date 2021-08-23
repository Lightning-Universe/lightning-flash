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
from typing import Any, Dict, List, Mapping, Optional, Type, Union

import torch
from torch.optim import Optimizer

from flash.core.adapter import AdapterTask
from flash.core.data.process import Serializer
from flash.core.registry import FlashRegistry
from flash.image.instance_segmentation.backbones import INSTANCE_SEGMENTATION_HEADS


class InstanceSegmentation(AdapterTask):
    """The ``InstanceSegmentation`` is a :class:`~flash.Task` for detecting objects in images. For more details, see
    :ref:`object_detection`.

    Args:
        num_classes: the number of classes for detection, including background
        model: a string of :attr`_models`. Defaults to 'fasterrcnn'.
        backbone: Pretained backbone CNN architecture. Constructs a model with a
            ResNet-50-FPN backbone when no backbone is specified.
        fpn: If True, creates a Feature Pyramind Network on top of Resnet based CNNs.
        pretrained: if true, returns a model pre-trained on COCO train2017
        pretrained_backbone: if true, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers: number of trainable resnet layers starting from final block.
            Only applicable for `fasterrcnn`.
        loss: the function(s) to update the model with. Has no effect for torchvision detection models.
        metrics: The provided metrics. All metrics here will be logged to progress bar and the respective logger.
            Changing this argument currently has no effect.
        optimizer: The optimizer to use for training. Can either be the actual class or the class name.
        pretrained: Whether the model from torchvision should be loaded with it's pretrained weights.
            Has no effect for custom models.
        learning_rate: The learning rate to use for training

    """

    heads: FlashRegistry = INSTANCE_SEGMENTATION_HEADS

    required_extras: str = "image"

    def __init__(
        self,
        num_classes: int,
        backbone: Optional[str] = "resnet18_fpn",
        head: Optional[str] = "mask_rcnn",
        pretrained: bool = True,
        optimizer: Type[Optimizer] = torch.optim.Adam,
        learning_rate: float = 5e-4,
        serializer: Optional[Union[Serializer, Mapping[str, Serializer]]] = None,
        **kwargs: Any,
    ):
        self.save_hyperparameters()

        metadata = self.heads.get(head, with_metadata=True)
        adapter = metadata["metadata"]["adapter"].from_task(
            self,
            num_classes=num_classes,
            backbone=backbone,
            head=head,
            pretrained=pretrained,
            **kwargs,
        )

        super().__init__(
            adapter,
            learning_rate=learning_rate,
            optimizer=optimizer,
            serializer=serializer,
        )

    def _ci_benchmark_fn(self, history: List[Dict[str, Any]]) -> None:
        """This function is used only for debugging usage with CI."""
        # todo
