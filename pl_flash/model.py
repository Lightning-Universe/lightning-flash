from typing import Callable, Mapping, Optional, Sequence, Type, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from pl_flash.utils import get_callable_dict


class LightningTask(pl.LightningModule):
    """A general LightningTask.

    Args:
        model: LightningTask to use for task.
        loss_fn: Loss function for training
        optimizer: Optimizer to use for training, defaults to `torch.optim.SGD`.
        metrics: Metrics to compute for training and evaluation.
        learning_rate: Learning rate to use for training, defaults to `1e-3`
    """

    _predict = False

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Optional[Union[Callable, Mapping, Sequence]],
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        metrics: Union[pl.metrics.Metric, Mapping, Sequence, None] = None,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_cls = optimizer
        self.metrics = nn.ModuleDict(get_callable_dict(metrics) if metrics is not None else {})
        self.loss_fn = {} if loss_fn is None else get_callable_dict(loss_fn)
        self.learning_rate = learning_rate
        # TODO: should we save more? Bug on some regarding yaml if we save metrics
        self.save_hyperparameters("learning_rate", "optimizer")

    def step(self, batch, batch_idx):
        """
        The training/validation/test step. Override for custom behavior.
        """
        output = {}
        if not self._predict:
            if isinstance(batch, dict):
                x, y = batch["x"], batch["target"]
            else:
                x, y = batch
        else:
            if isinstance(batch, dict):
                x = batch["x"]
            else:
                x, y = batch

        y_hat = self.forward(x)

        if self._predict:
            return self.output_to_metric(y_hat)

        output["y_hat"] = self.output_to_metric(y_hat)
        losses = {name: l_fn(y_hat, y) for name, l_fn in self.loss_fn.items()}
        logs = {}
        for name, metric in self.metrics.items():
            if isinstance(metric, pl.metrics.Metric):
                metric(self.output_to_metric(y_hat), y)
                logs[name] = metric  # log the metric itself if it is of type Metric
            else:
                logs[name] = metric(y_hat, y)
        logs.update(losses)
        if len(losses.values()) > 1:
            logs["total_loss"] = sum(losses.values())
            return logs["total_loss"], logs
        output["loss"] = list(losses.values())[0]
        output["logs"] = logs
        output["y"] = y
        return output

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        output = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in output["logs"].items()}, on_step=True, on_epoch=True, prog_bar=True)
        return output["loss"]

    def validation_step(self, batch, batch_idx):
        output = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in output["logs"].items()}, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        output = self.step(batch, batch_idx)
        if self._predict:
            if getattr(self, "add_predictions", None) is not None:
                self.add_predictions(output)
            return output

        self.log_dict({f"test_{k}": v for k, v in output["logs"].items()}, on_step=False, on_epoch=True, prog_bar=True)

        # ``pip install https://github.com/PyTorchLightning/pytorch-lightning/archive/flash_inference.zip``
        if getattr(self, "add_predictions", None) is not None:
            self.add_predictions([{"target": y, "pred": y_hat} for y, y_hat in zip(output["y"], output["y_hat"])])

    def configure_optimizers(self):
        return self.optimizer_cls(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def output_to_metric(output):
        return output


class ClassificationLightningTask(LightningTask):

    @staticmethod
    def output_to_metric(output):
        return F.softmax(output, -1)
