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
from torch import nn

from flash.core.data import DataModule
from flash.core.utils import get_callable_dict
from flash.data.data_pipeline import DataPipeline
from flash.data.postprocessing_pipeline import PostProcessingPipeline


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
        if isinstance(x, str) and os.path.isdir(x):
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
        return self._get_pipeline('data')

    @data_pipeline.setter
    def data_pipeline(self, data_pipeline: DataPipeline) -> None:
        self._data_pipeline = data_pipeline

    @property
    def postprocessing_pipeline(self) -> PostProcessingPipeline:
        return self._get_pipeline('postprocessing')

    def _get_pipeline(self, pipeline_type: str):
        pipeline_attr_name = f'{pipeline_type}_pipline'

        if getattr(self, '_' + pipeline_attr_name) is not None:
            return getattr(self, '_' + pipeline_attr_name)

        if self.datamodule is not None and hasattr(self, pipeline_attr_name):
            return getattr(self.datamodule, pipeline_attr_name)

        if self.trainer is not None and hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
            if hasattr(self.trainer.datamodule,
                       pipeline_attr_name) and getattr(self.trainer.datamodule, pipeline_attr_name is not None):
                return getattr(self.trainer.datamodule, pipeline_attr_name is not None)

        return None

    @staticmethod
    def default_data_pipeline() -> DataPipeline:
        """Pipeline to use when there is no datamodule or it has not defined its pipeline"""
        return DataModule.default_data_pipeline()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.data_pipeline = checkpoint["pipeline"]

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["pipeline"] = self.data_pipeline

    def configure_finetune_callback(self):
        return []

    ### THE FOLLOWING IS A POC FOR DISTRIBUTED PREDICTION
    def on_predict_start(self):
        # TODO: Add hook to lightning Trainer
        if self.data_pipeline is not None:
            self.data_pipeline._attach_to_model(self)

        if self.postprocessing_pipeline is not None:
            self.postprocessing_pipeline._attach_to_model(self)

    def predict_step(self, batch, batch_idx):
        # TODO: Move lightning predict loop from predict to predict_step
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x, y = batch
        else:
            x, y = batch, None

        return self(x)

    def new_predict(
        self,
        x: Any,
        skip_collate: Optional[bool] = None,
        data_pipeline: Optional[DataPipeline] = None,
        postprocessing_pipeline: Optional[PostProcessingPipeline] = None,
        data_loader_kwargs: Optional[dict] = None,
        **trainer_kwargs
    ):
        if data_pipeline is not None:
            self.data_pipeline = data_pipeline
        if postprocessing_pipeline is not None:
            self.postprocessing_pipeline = postprocessing_pipeline

        trainer = self._create_trainer('predict', **trainer_kwargs)

        if data_loader_kwargs is None:
            data_loader_kwargs = {}

        if 'num_workers' not in data_loader_kwargs:
            # leave one for main process
            data_loader_kwargs['num_workers'] = os.cpu_count() - 1

        auto_collate = None
        if 'collate_fn' not in data_loader_kwargs:
            auto_collate = not skip_collate

        dl = self.data_pipeline._generate_loader(x, auto_collate=auto_collate, **data_loader_kwargs)

        return trainer.predict(self, dl)

    def _create_trainer(self, stage: str, **trainer_kwargs):
        # TODO: Also use these for trainer creation in training?
        # TODO: Have default trainer kwargs per task?
        _trainer_kwargs = {}
        # TODO: Adjust this to trainer running stage from pl
        if stage == 'predict':
            _trainer_kwargs.update(logger=None)

        if not 'gpus' in trainer_kwargs and not 'tpu_cores' in trainer_kwargs:
            _trainer_kwargs['gpus'], _trainer_kwargs['tpu_cores'] = self._parse_default_devices()

        _trainer_kwargs.update(trainer_kwargs)

        if not hasattr(self, 'trainer') or self.trainer is None or self._last_trainer_kwargs != trainer_kwargs:
            self._last_trainer_kwargs = _trainer_kwargs
            self.trainer = None
            return Trainer(**_trainer_kwargs)

        else:
            return self.trainer

    def _parse_default_devices(self):
        gpus = None,
        tpu_cores = None

        if torch.cuda.is_available():
            gpus = torch.cuda.device_count()

        # TODO: Add logic for automatted TPU device parsing

        return gpus, tpu_cores

    def serve(
        self,
        x,
        skip_collate: Optional[bool] = None,
        data_pipeline: Optional[DataPipeline] = None,
        postprocessing_pipeline: Optional[PostProcessingPipeline] = None,
        data_loader_kwargs: Optional[dict] = None,
        **trainer_kwargs
    ):
        """Serving for Production. Basically same as prediction, just other defaults (no workers, no distributed prediction)
        """

        if data_loader_kwargs is None:
            data_loader_kwargs = {}
        data_loader_kwargs['num_workers'] = 0

        trainer_kwargs['num_gpus'] = [0] if torch.cuda.is_available() else 0
        # TODO: tpu_cores
        return self.new_predict(
            x,
            skip_collate=skip_collate,
            data_pipeline=data_pipeline,
            postprocessing_pipeline=postprocessing_pipeline,
            data_loader_kwargs=data_loader_kwargs,
            **trainer_kwargs
        )
