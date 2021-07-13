from types import FunctionType
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import torch
from torch import nn
from torchmetrics import Accuracy, F1, Metric

from flash.core.classification import ClassificationTask
from flash.core.data.data_source import DefaultDataKeys
from flash.core.registry import FlashRegistry

IMAGE_CLASSIFIER_STRATEGIES = FlashRegistry("strategies")


@IMAGE_CLASSIFIER_STRATEGIES(name="supervised")
class SupervisedTrainingStrategy(ClassificationTask):

    def __init__(
        self,
        backbone,
        num_features: int,
        num_classes: int,
        head: Optional[Union[FunctionType, nn.Module]] = None,
        loss_fn: Optional[Callable] = None,
        metrics: Union[Metric, Callable, Mapping, Sequence, None] = None,
        multi_label: bool = False,
    ):
        super().__init__(
            model=None,
            loss_fn=loss_fn,
            metrics=metrics or F1(num_classes) if multi_label else Accuracy(),
            multi_label=multi_label,
        )

        self.backbone = backbone

        head = head(num_features, num_classes) if isinstance(head, FunctionType) else head
        self.head = head or nn.Linear(num_features, num_classes)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return super().test_step(batch, batch_idx)

    def forward(self, x) -> torch.Tensor:
        x = self.backbone(x)
        if x.dim() == 4:
            x = x.mean(-1).mean(-1)
        return self.head(x)
