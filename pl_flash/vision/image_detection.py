from typing import Any, Union, Mapping, Sequence, Type, Callable

import torch
from torch import nn
from torch.optim import Optimizer
import torchvision

from pl_flash.core import Flash
from pytorch_lightning import TrainResult


class ImageDetector(Flash):
    """Image detection task

    Args:
        num_classes (int): the number of classes for detection, including background
        model (string or nn.Module): either a string of :attr`available_models` or a custom nn.Module.
            Defaults to 'fasterrcn'.
        loss: the function(s) to update the model with. Has no effect for torchvision detection models.
        metrics: The provided metrics. All metrics here will be logged to progress bar and the respective logger.
            Defaults to None.
        optimizer: The optimizer to use for training. Can either be the actual class or the class name.
            Defaults to Adam.
        pretrained: Whether the model from torchvision or bolts should be loaded with it's pretrained weights.
            Has no effect for custom models. Defaults to True.
    """

    _available_models_torchvision = ("fasterrcnn_resnet50_fpn",)
    _available_models_bolts = ()
    available_models: tuple = tuple(list(_available_models_torchvision) + list(_available_models_bolts))

    def __init__(
        self,
        num_classes: int,
        model: Union[str, nn.Module] = "fasterrcnn_resnet50_fpn",
        loss=None,
        metrics: Union[Callable, nn.Module, Mapping, Sequence, None] = None,
        optimizer: Union[Type[Optimizer], str] = "Adam",
        pretrained: bool = True,
        learning_rate=1e-3,
        **kwargs,
    ):
        if isinstance(model, str):
            assert model in self.available_models

        if loss is None:
            # TODO: maybe better way of handling no loss,
            loss = {}
        super().__init__(
            model=model, loss=loss, metrics=metrics, learning_rate=learning_rate, optimizer=optimizer,
        )

        self.num_classes = num_classes
        if isinstance(self.model, str) and self.model in self._available_models_torchvision:
            self.model = self._model_from_torchvision(model, pretrained, num_classes, **kwargs)

    @staticmethod
    def _model_from_torchvision(model: str, pretrained: bool, num_classes: int, **kwargs):
        model = getattr(torchvision.models.detection, model)(pretrained=pretrained, **kwargs)

        if isinstance(model, torchvision.models.detection.FasterRCNN):
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            head = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
            model.roi_heads.box_predictor = head
            return model

    def training_step(self, batch, batch_idx):
        """The training step.
        Overrides Flash.training_step
        """
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]

        # fasterrcnn takes both images and targets for training, returns loss_dict
        loss_dict = self.model(images, targets)
        loss = sum(loss_dict.values())
        result = TrainResult(loss)
        for k, v in loss_dict.items():
            result.log(k, v)

        return result
