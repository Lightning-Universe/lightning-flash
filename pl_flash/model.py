from typing import Callable, Mapping, Sequence, Union, Type

import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl

from pl_flash.utils import get_callable_dict


class Model(pl.LightningModule):
    """A general Task.

    Args:
        model: Model to use for task.
        loss_fn: Loss function for training
        optimizer: Optimizer to use for training, defaults to `torch.optim.SGD`.
        metrics: Metrics to compute for training and evaluation.
        learning_rate: Learning rate to use for training, defaults to `1e-3`
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Union[Callable, Mapping, Sequence],
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        metrics: Union[Callable, Mapping, Sequence, None] = None,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_cls = optimizer
        self.metrics = get_callable_dict(metrics) if metrics is not None else {}
        self.loss_fn = get_callable_dict(loss_fn)
        self.learning_rate = learning_rate
        # TODO: should we save more? Bug on some regarding yaml if we save metrics
        self.save_hyperparameters("learning_rate", "optimizer")

    def step(self, batch, batch_idx):
        """
        The training/validation/test step. Override for custom behavior.
        """
        x, y = batch
        y_hat = self.forward(x)
        losses = {name: l_fn(y_hat, y) for name, l_fn in self.loss_fn.items()}
        logs = {}
        for name, metric in self.metrics.items():
            if isinstance(metric, pl.metrics.Metric):
                metric(y_hat, y)
                logs[name] = metric  # log the metric itself if it is of type Metric
            else:
                logs[name] = metric(y_hat, y)
        logs.update(losses)
        logs["total_loss"] = sum(losses.values())
        return logs["total_loss"], logs

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})

    def test_step(self, batch, batch_idx):
        _, logs = self.step(batch, batch_idx)
        self.log_dict({f"test_{k}": v for k, v in logs.items()})

    def configure_optimizers(self):
        return self.optimizer_cls(self.parameters(), lr=self.learning_rate)
