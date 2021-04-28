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
from types import FunctionType
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Type, Union

import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy, IoU

from flash.core.classification import ClassificationTask
from flash.core.registry import FlashRegistry
from flash.data.process import Preprocess, Serializer
from flash.utils.imports import _TIMM_AVAILABLE, _TORCHVISION_AVAILABLE
from flash.vision.segmentation.serialization import SegmentationLabels

if _TORCHVISION_AVAILABLE:
    import torchvision

SEMANTIC_SEGMENTATION_BACKBONES = FlashRegistry("backbones")


class SemanticSegmentation(ClassificationTask):
    """Task that performs semantic segmentation on images.

    Use a built in backbone

    Example::

        from flash.vision import SemanticSegmentation

        segmentation = SemanticSegmentation(
            num_classes=21, backbone="torchvision/fcn_resnet50"
        )

    Args:
        num_classes: Number of classes to classify.
        backbone: A string or (model, num_features) tuple to use to compute image features, defaults to ``"torchvision/fcn_resnet50"``.
        backbone_kwargs: Additional arguments for the backbone configuration.
        pretrained: Use a pretrained backbone, defaults to ``False``.
        loss_fn: Loss function for training, defaults to :func:`torch.nn.functional.cross_entropy`.
        optimizer: Optimizer to use for training, defaults to :class:`torch.optim.Adam`.
        metrics: Metrics to compute for training and evaluation, defaults to :class:`torchmetrics.IoU`.
        learning_rate: Learning rate to use for training, defaults to ``1e-3``.
        multi_label: Whether the targets are multi-label or not.
        serializer: The :class:`~flash.data.process.Serializer` to use when serializing prediction outputs.
    """

    backbones: FlashRegistry = SEMANTIC_SEGMENTATION_BACKBONES

    def __init__(
        self,
        num_classes: int,
        backbone: Union[str, Tuple[nn.Module, int]] = "torchvision/fcn_resnet50",
        backbone_kwargs: Optional[Dict] = None,
        pretrained: bool = True,
        loss_fn: Optional[Callable] = None,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        metrics: Optional[Union[Callable, Mapping, Sequence, None]] = None,
        learning_rate: float = 1e-3,
        multi_label: bool = False,
        serializer: Optional[Union[Serializer, Mapping[str, Serializer]]] = None,
    ) -> None:

        if metrics is None:
            metrics = IoU(num_classes=num_classes)

        if loss_fn is None:
            loss_fn = F.cross_entropy

        # TODO: need to check for multi_label
        if multi_label:
            raise NotImplementedError("Multi-label not supported yet.")

        super().__init__(
            model=None,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,
            serializer=serializer or SegmentationLabels(),
        )

        self.save_hyperparameters()

        if not backbone_kwargs:
            backbone_kwargs = {}

        # TODO: pretrained to True causes some issues
        self.backbone = self.backbones.get(backbone)(pretrained=pretrained, num_classes=num_classes, **backbone_kwargs)

    def forward(self, x) -> torch.Tensor:
        return self.backbone(x)['out']  # TODO: find a proper way to get 'out' from registry


@SemanticSegmentation.backbones(name="torchvision/fcn_resnet50")
def load_torchvision_fcn_resnet50(pretrained: bool, num_classes: int) -> nn.Module:
    model = torchvision.models.segmentation.fcn_resnet50(pretrained=pretrained)
    model.classifier[-1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model


@SemanticSegmentation.backbones(name="torchvision/fcn_resnet101")
def load_torchvision_fcn_resnet101(pretrained: bool, num_classes: int) -> nn.Module:
    model = torchvision.models.segmentation.fcn_resnet101(pretrained=pretrained)
    model.classifier[-1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model
