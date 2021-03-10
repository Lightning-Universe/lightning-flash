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
from pytorch_lightning import Trainer
from pytorch_lightning.trainer.states import RunningStage, TrainerState
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn

from flash.core.utils import get_callable_dict
from flash.data.data_pipeline import DataPipeline, Postprocess, Preprocess


def predict_context(func: Callable) -> Callable:
    """
    This decorator is used as context manager
    to put model in eval mode before running predict and reset to train after.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs) -> Any:
        grad_enabled = torch.is_grad_enabled()
        is_training = self.training
        self.eval()
        torch.set_grad_enabled(False)

        result = func(self, *args, **kwargs)

        if is_training:
            self.train()
        torch.set_grad_enabled(grad_enabled)
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
        self._last_trainer_kwargs = {}

        if model is not None:
            self.model = model
        self.loss_fn = {} if loss_fn is None else get_callable_dict(loss_fn)
        self.optimizer_cls = optimizer
        self.metrics = nn.ModuleDict({} if metrics is None else get_callable_dict(metrics))
        self.learning_rate = learning_rate
        # TODO: should we save more? Bug on some regarding yaml if we save metrics
        self.save_hyperparameters("learning_rate", "optimizer")

        if not hasattr(self, "_data_pipeline"):
            self._data_pipeline = None
        if not hasattr(self, "_preprocess"):
            self._preprocess = None
        if not hasattr(self, "_postprocess"):
            self._postprocess = None

    def step(self, batch: Any, batch_idx: int) -> Any:
        """
        The training/validation/test step. Override for custom behavior.
        """
        x, y = batch
        y_hat = self.forward(x)
        output = {"y_hat": self.postprocess.per_batch_transform(y_hat)}
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
        running_stage = RunningStage.PREDICTING
        data_pipeline = data_pipeline or self.data_pipeline
        x = [x for x in data_pipeline._generate_auto_dataset(x, running_stage)]
        x = data_pipeline.worker_preprocessor(running_stage)(x)
        x = self.transfer_batch_to_device(x, self.device)
        x = data_pipeline.device_preprocessor(running_stage)(x)
        predictions = self.predict_step(x, 0)
        predictions = data_pipeline.postprocessor(predictions)
        return predictions

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        if isinstance(batch, tuple):
            batch = batch[0]
        elif isinstance(batch, list):
            # Todo: Understand why stack is needed
            batch = torch.stack(batch)
        return self(batch)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer_cls(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.data_pipeline = checkpoint["pipeline"]

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["pipeline"] = self.data_pipeline

    def configure_finetune_callback(self):
        return []

    @property
    def preprocess(self):
        return self._preprocess

    @preprocess.setter
    def preprocess(self, preprocess: Preprocess) -> None:
        data_pipeline = self.data_pipeline
        self.data_pipeline = DataPipeline(preprocess, data_pipeline._postprocess_pipeline or self._postprocess)
        import pdb
        pdb.set_trace()

    @property
    def postprocess(self):
        return self._postprocess

    @postprocess.setter
    def postprocess(self, postprocess: Postprocess) -> None:
        data_pipeline = self.data_pipeline
        self.data_pipeline = DataPipeline(data_pipeline._preprocess_pipeline, postprocess)
        self._preprocess = self.data_pipeline._preprocess_pipeline
        self._postprocess = self.data_pipeline._postprocess_pipeline

    @property
    def data_pipeline(self) -> Optional[DataPipeline]:
        # we need to save the pipeline in case this class
        # is loaded from checkpoint and used to predict
        if self._data_pipeline is not None:
            return self._data_pipeline
        self.data_pipeline = self._get_pipeline("data_pipeline")
        return self._data_pipeline

    @data_pipeline.setter
    def data_pipeline(self, data_pipeline: DataPipeline) -> None:
        if not isinstance(data_pipeline, DataPipeline):
            raise MisconfigurationException(f"Excepted to receive a DataPipeline. Found {data_pipeline}")
        self._data_pipeline = DataPipeline(data_pipeline.preprocess, self.postprocess)
        self._data_pipeline._attach_to_model(self)

        if self._preprocess is not None or self._postprocess is not None:
            return DataPipeline(self._preprocess, self._postprocess)

        if self.datamodule is not None and getattr(self.datamodule, 'data_pipeline', None) is not None:
            return self.datamodule.data_pipeline

        if self.trainer is not None and hasattr(
            self.trainer, 'datamodule'
        ) and getattr(self.trainer.datamodule, 'data_pipeline', None) is not None:
            return self.trainer.datamodule.data_pipeline
        return self._data_pipeline

    @data_pipeline.setter
    def data_pipeline(self, data_pipeline: DataPipeline) -> None:
        self._data_pipeline = data_pipeline
        if data_pipeline is not None and getattr(data_pipeline, '_preprocess_pipeline', None) is not None:
            self._preprocess = data_pipeline._preprocess_pipeline

        if data_pipeline is not None and getattr(data_pipeline, '_postprocess_pipeline', None) is not None:
            datapipeline_postprocess = getattr(data_pipeline, '_postprocess_pipeline', None)
            if type(datapipeline_postprocess) != Postprocess:
                self._postprocess = data_pipeline._postprocess_pipeline

    def on_train_start(self) -> None:
        if self.data_pipeline is not None:
            self.data_pipeline._attach_to_model(self, RunningStage.TRAINING)
        return super().on_train_start()

    def on_train_end(self) -> None:
        if self.data_pipeline is not None:
            self.data_pipeline._detach_from_model(self)
        return super().on_train_end()

    def on_validation_start(self) -> None:
        if self.data_pipeline is not None:
            self.data_pipeline._detach_from_model(self)
        if self.data_pipeline is not None:
            self.data_pipeline._attach_to_model(self, RunningStage.VALIDATING)
        return super().on_validation_start()

    def on_validation_end(self) -> None:
        if self.data_pipeline is not None:
            self.data_pipeline._detach_from_model(self)
        if self.trainer.state == TrainerState.FITTING:
            self.data_pipeline._attach_to_model(self, RunningStage.TRAINING)
        return super().on_validation_end()

    def on_test_start(self) -> None:
        if self.data_pipeline is not None:
            self.data_pipeline._attach_preprocess_to_model(self, RunningStage.TESTING)
        return super().on_test_start()

    def on_test_end(self):
        if self.data_pipeline is not None:
            self.data_pipeline._detach_from_model(self)
        return super().on_test_end()

    def on_predict_start(self):
        if self.data_pipeline is not None:
            self.data_pipeline._attach_to_model(self, RunningStage.PREDICTING)

        return super().on_predict_start()

    def on_predict_end(self):
        if self.data_pipeline is not None:
            self.data_pipeline._detach_from_model(self)
        return super().on_predict_end()
