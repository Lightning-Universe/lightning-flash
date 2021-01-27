from typing import Callable, Mapping, Optional, Sequence, Type, Union, Any

import pytorch_lightning as pl
import torch
from torch import nn

from flash.core.data import DataPipeline
from flash.core.utils import get_callable_dict


class Task(pl.LightningModule):
    """A general Task.

    Args:
        model: Model to use for the task.
        loss_fn: Loss function for training
        optimizer: Optimizer to use for training, defaults to `torch.optim.SGD`.
        metrics: Metrics to compute for training and evaluation.
        learning_rate: Learning rate to use for training, defaults to `1e-3`
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Optional[Union[Callable, Mapping, Sequence]] = None,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        metrics: Union[pl.metrics.Metric, Mapping, Sequence, None] = None,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = {} if loss_fn is None else get_callable_dict(loss_fn)
        self.optimizer_cls = optimizer
        self.metrics = nn.ModuleDict({} if metrics is None else get_callable_dict(metrics))
        self.learning_rate = learning_rate
        # TODO: should we save more? Bug on some regarding yaml if we save metrics
        self.save_hyperparameters("learning_rate", "optimizer")

    def step(self, batch: Any, batch_idx: int):
        """
        The training/validation/test step. Override for custom behavior.
        """
        x, y = (batch["x"], batch["target"]) if isinstance(batch, dict) else batch
        y_hat = self.forward(x)
        output = {"y_hat": self.data_pipeline.before_uncollate(y_hat)}
        losses = {name: l_fn(y_hat, y) for name, l_fn in self.loss_fn.items()}
        logs = {}
        for name, metric in self.metrics.items():
            if isinstance(metric, pl.metrics.Metric):
                metric(output["y_hat"], y)
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

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int):
        output = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in output["logs"].items()}, on_step=True, on_epoch=True, prog_bar=True)
        return output["loss"]

    def validation_step(self, batch: Any, batch_idx: int):
        output = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in output["logs"].items()}, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        output = self.step(batch, batch_idx)
        self.log_dict({f"test_{k}": v for k, v in output["logs"].items()}, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer_cls(self.parameters(), lr=self.learning_rate)

    @property
    def data_pipeline(self) -> DataPipeline:
        try:
            return self.trainer.datamodule.data_pipeline
        except AttributeError:
            return self.default_pipeline

    @property
    def default_pipeline(self) -> DataPipeline:
        """Pipeline to use when there is no datamodule or it has not defined its pipeline"""
        return DataPipeline()
