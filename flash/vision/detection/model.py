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
from typing import Any, Callable, Mapping, Sequence, Type, Union

import torch
import torchvision
from torch import nn
from torch.optim import Optimizer

from flash.core import Task

_models = {"fasterrcnn_resnet50_fpn": torchvision.models.detection.fasterrcnn_resnet50_fpn}


class ImageDetector(Task):
    """Image detection task

    Args:
        num_classes: the number of classes for detection, including background
        model: either a string of :attr`_models` or a custom nn.Module.
            Defaults to 'fasterrcnn_resnet50_fpn'.
        loss: the function(s) to update the model with. Has no effect for torchvision detection models.
        metrics: The provided metrics. All metrics here will be logged to progress bar and the respective logger.
            Defaults to None.
        optimizer: The optimizer to use for training. Can either be the actual class or the class name.
            Defaults to Adam.
        pretrained: Whether the model from torchvision should be loaded with it's pretrained weights.
            Has no effect for custom models. Defaults to True.
        learning_rate: The learning rate to use for training

    """

    def __init__(
        self,
        num_classes: int,
        model: Union[str, nn.Module] = "fasterrcnn_resnet50_fpn",
        loss=None,
        metrics: Union[Callable, nn.Module, Mapping, Sequence, None] = None,
        optimizer: Type[Optimizer] = torch.optim.Adam,
        pretrained: bool = True,
        learning_rate=1e-3,
        **kwargs,
    ):
        if model in _models:
            model = _models[model](pretrained=pretrained)
            if isinstance(model, torchvision.models.detection.FasterRCNN):
                in_features = model.roi_heads.box_predictor.cls_score.in_features
                head = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
                model.roi_heads.box_predictor = head

        loss = {} if loss is None else loss

        super().__init__(
            model=model,
            loss_fn=loss,
            metrics=metrics,
            learning_rate=learning_rate,
            optimizer=optimizer,
        )

    def step(self, batch: Any, batch_idx: int) -> Any:
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]

        # fasterrcnn takes both images and targets for training, returns loss_dict
        loss_dict = self.model(images, targets)
        return loss_dict

    def training_step(self, batch, batch_idx) -> Any:
        """The training step.
        Overrides Task.training_step
        """
        loss_dict = self.step(batch, batch_idx)
        loss = sum(loss_dict.values())
        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        loss_dict = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        loss_dict = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True, prog_bar=True)
