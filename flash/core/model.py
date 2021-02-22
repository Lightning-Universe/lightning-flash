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
import functools
import os
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Type, Union

import pytorch_lightning as pl
import torch
from torch import nn

from flash.core.data import DataModule, DataPipeline
from flash.core.utils import get_callable_dict


def predict_context(func: Callable) -> Callable:
    """
    This decorator is used as context manager
    to put model in eval mode before running predict and reset to train after.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs) -> Any:
        self.eval()
        torch.set_grad_enabled(False)

        result = func(self, *args, **kwargs)

        self.train()
        torch.set_grad_enabled(True)
        return result

    return wrapper


class Task(pl.LightningModule):
    """A general Task.

    Args:
        model: Model to use for the task.
        loss_fn: Loss function for training
        optimizer: Optimizer to use for training, defaults to `torch.optim.SGD`.
        metrics: Metrics to compute for training and evaluation.
        learning_rate: Learning rate to use for training, defaults to `5e-5`
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        loss_fn: Optional[Union[Callable, Mapping, Sequence]] = None,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        metrics: Union[pl.metrics.Metric, Mapping, Sequence, None] = None,
        learning_rate: float = 5e-5,
    ):
        super().__init__()
        if model is not None:
            self.model = model
        self.loss_fn = {} if loss_fn is None else get_callable_dict(loss_fn)
        self.optimizer_cls = optimizer
        self.metrics = nn.ModuleDict({} if metrics is None else get_callable_dict(metrics))
        self.learning_rate = learning_rate
        # TODO: should we save more? Bug on some regarding yaml if we save metrics
        self.save_hyperparameters("learning_rate", "optimizer")
        self._data_pipeline = None

    def step(self, batch: Any, batch_idx: int) -> Any:
        """
        The training/validation/test step. Override for custom behavior.
        """
        x, y = batch
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

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        output = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in output["logs"].items()}, on_step=True, on_epoch=True, prog_bar=True)
        return output["loss"]

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        output = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in output["logs"].items()}, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        output = self.step(batch, batch_idx)
        self.log_dict({f"test_{k}": v for k, v in output["logs"].items()}, on_step=False, on_epoch=True, prog_bar=True)

    @predict_context
    def predict(
        self,
        x: Any,
        batch_idx: Optional[int] = None,
        skip_collate_fn: bool = False,
        dataloader_idx: Optional[int] = None,
        data_pipeline: Optional[DataPipeline] = None,
    ) -> Any:
        """
        Predict function for raw data or processed data

        Args:

            x: Input to predict. Can be raw data or processed data. If str, assumed to be a folder of data.

            batch_idx: Batch index

            dataloader_idx: Dataloader index

            skip_collate_fn: Whether to skip the collate step.
                this is required when passing data already processed
                for the model, for example, data from a dataloader

            data_pipeline: Use this to override the current data pipeline

        Returns:
            The post-processed model predictions

        """
        # enable x to be a path to a folder
        if isinstance(x, str):
            files = os.listdir(x)
            files = [os.path.join(x, y) for y in files]
            x = files

        data_pipeline = data_pipeline or self.data_pipeline
        batch = x if skip_collate_fn else data_pipeline.collate_fn(x)
        batch_x, batch_y = batch if len(batch) == 2 and isinstance(batch, (list, tuple)) else (batch, None)
        predictions = self.forward(batch_x)
        output = data_pipeline.uncollate_fn(predictions)  # TODO: pass batch and x
        return output

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer_cls(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)

    @property
    def data_pipeline(self) -> DataPipeline:
        # we need to save the pipeline in case this class
        # is loaded from checkpoint and used to predict
        if not self._data_pipeline:
            try:
                # datamodule pipeline takes priority
                self._data_pipeline = self.trainer.datamodule.data_pipeline
            except AttributeError:
                self._data_pipeline = self.default_pipeline()
        return self._data_pipeline

    @data_pipeline.setter
    def data_pipeline(self, data_pipeline: DataPipeline) -> None:
        self._data_pipeline = data_pipeline

    @staticmethod
    def default_pipeline() -> DataPipeline:
        """Pipeline to use when there is no datamodule or it has not defined its pipeline"""
        return DataModule.default_pipeline()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.data_pipeline = checkpoint["pipeline"]

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["pipeline"] = self.data_pipeline

    def configure_finetune_callback(self):
        return []
