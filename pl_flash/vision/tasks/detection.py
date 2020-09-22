from typing import Any, Optional, Union, Mapping, Sequence, Type, Callable

import torch
from torch import nn
from torch.optim import Optimizer

from pytorch_lightning import TrainResult

from pl_flash.flash import Flash

__all__ = ["ImageDetector"]


class ImageDetector(Flash):
    """Image detection task

    Args:
        num_classes: the number of classes for detection, including background
        model: either a string of :attr`available_models` or a custom nn.Module.
            Defaults to 'fasterrcn'.
        loss: the function(s) to update the model with. Has no effect for torchvision detection models.
        metrics: The provided metrics. All metrics here will be logged to progress bar and the respective logger.
            Defaults to None.
        optimizer: The optimizer to use for training. Can either be the actual class or the class name.
            Defaults to Adam.
        pretrained: Whether the model from torchvision or bolts should be loaded with it's pretrained weights.
            Has no effect for custom models. Defaults to True.
        learing_rate: The learning rate to use for training
    """

    _available_models_torchvision = ("fasterrcnn_resnet50_fpn",)
    _available_models_bolts = ()
    available_models: tuple = tuple(list(_available_models_torchvision) + list(_available_models_bolts))

    def __init__(
        self,
        model: Union[str, nn.Module] = "fasterrcnn_resnet50_fpn",
        num_classes: Optional[int] = None,
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
            model=model,
            loss=loss,
            metrics=metrics,
            learning_rate=learning_rate,
            optimizer=optimizer,
        )

        self.num_classes = num_classes
        if isinstance(self.model, str) and self.model in self._available_models_torchvision:
            self.model = self._model_from_torchvision(model, pretrained, num_classes, **kwargs)

    @staticmethod
    def _model_from_torchvision(model: str, pretrained: bool, num_classes: int, **kwargs) -> torch.nn.Module:
        """Retrieve a model from torchvision

        Args:
            model: the model to retrieve from torchvision
            pretrained: whether to also load pretrained weights
            num_classes: the number of classes of the final model

        Raises:
            ImportError: torchvision is not installed
            TypeError: unexpected model type

        Returns:
            torch.nn.Module: the retrieved model
        """
        try:
            import torchvision.models.detection

        except ImportError as e:
            raise ImportError(
                "Torchvision is not installed please install it following the guides on"
                + "https://pytorch.org/get-started/locally/"
            ) from e

        assert num_classes is not None

        model = getattr(torchvision.models.detection, model)(pretrained=pretrained, **kwargs)

        if isinstance(model, torchvision.models.detection.FasterRCNN):
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            head = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
            model.roi_heads.box_predictor = head
            return model

        raise TypeError

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
