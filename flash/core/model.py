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
import inspect
from copy import deepcopy
from importlib import import_module
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

import flash
from flash.core.data.data_pipeline import DataPipeline, DataPipelineState
from flash.core.data.data_source import DataSource
from flash.core.data.process import Postprocess, Preprocess, Serializer, SerializerMapping
from flash.core.registry import FlashRegistry
from flash.core.schedulers import _SCHEDULERS_REGISTRY
from flash.core.utilities.apply_func import get_callable_dict


class BenchmarkConvergenceCI(Callback):

    def __init__(self):
        self.history = []

    def on_validation_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        self.history.append(deepcopy(trainer.callback_metrics))
        if trainer.current_epoch == trainer.max_epochs - 1:
            fn = getattr(pl_module, "_ci_benchmark_fn", None)
            if fn:
                fn(self.history)
                if trainer.is_global_zero:
                    print("Benchmark Successful!")


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


class Task(LightningModule):
    """A general Task.

    Args:
        model: Model to use for the task.
        loss_fn: Loss function for training
        optimizer: Optimizer to use for training, defaults to :class:`torch.optim.Adam`.
        metrics: Metrics to compute for training and evaluation.
        learning_rate: Learning rate to use for training, defaults to ``5e-5``.
        preprocess: :class:`~flash.core.data.process.Preprocess` to use as the default for this task.
        postprocess: :class:`~flash.core.data.process.Postprocess` to use as the default for this task.
    """

    schedulers: FlashRegistry = _SCHEDULERS_REGISTRY

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        loss_fn: Optional[Union[Callable, Mapping, Sequence]] = None,
        optimizer: Union[Type[torch.optim.Optimizer], torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Union[Type[_LRScheduler], str, _LRScheduler]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        metrics: Union[torchmetrics.Metric, Mapping, Sequence, None] = None,
        learning_rate: float = 5e-5,
        preprocess: Optional[Preprocess] = None,
        postprocess: Optional[Postprocess] = None,
        serializer: Optional[Union[Serializer, Mapping[str, Serializer]]] = None,
    ):
        super().__init__()
        if model is not None:
            self.model = model
        self.loss_fn = {} if loss_fn is None else get_callable_dict(loss_fn)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_kwargs = scheduler_kwargs or {}

        self.metrics = nn.ModuleDict({} if metrics is None else get_callable_dict(metrics))
        self.learning_rate = learning_rate
        # TODO: should we save more? Bug on some regarding yaml if we save metrics
        self.save_hyperparameters("learning_rate", "optimizer")

        self._preprocess: Optional[Preprocess] = preprocess
        self._postprocess: Optional[Postprocess] = postprocess
        self._serializer: Optional[Serializer] = None

        # TODO: create enum values to define what are the exact states
        self._data_pipeline_state: Optional[DataPipelineState] = None

        # Explicitly set the serializer to call the setter
        self.serializer = serializer

    def step(self, batch: Any, batch_idx: int) -> Any:
        """
        The training/validation/test step. Override for custom behavior.
        """
        x, y = batch
        y_hat = self(x)
        output = {"y_hat": y_hat}
        losses = {name: l_fn(y_hat, y) for name, l_fn in self.loss_fn.items()}
        logs = {}
        y_hat = self.to_metrics_format(y_hat)
        for name, metric in self.metrics.items():
            if isinstance(metric, torchmetrics.metric.Metric):
                metric(y_hat, y)
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

    def to_metrics_format(self, x: torch.Tensor) -> torch.Tensor:
        return x

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
        data_source: Optional[str] = None,
        data_pipeline: Optional[DataPipeline] = None,
    ) -> Any:
        """
        Predict function for raw data or processed data

        Args:
            x: Input to predict. Can be raw data or processed data. If str, assumed to be a folder of data.

            data_pipeline: Use this to override the current data pipeline

        Returns:
            The post-processed model predictions
        """
        running_stage = RunningStage.PREDICTING

        data_pipeline = self.build_data_pipeline(data_source or "default", data_pipeline)

        x = [x for x in data_pipeline.data_source.generate_dataset(x, running_stage)]
        x = data_pipeline.worker_preprocessor(running_stage)(x)
        # switch to self.device when #7188 merge in Lightning
        x = self.transfer_batch_to_device(x, next(self.parameters()).device)
        x = data_pipeline.device_preprocessor(running_stage)(x)
        predictions = self.predict_step(x, 0)  # batch_idx is always 0 when running with `model.predict`
        predictions = data_pipeline.postprocessor(running_stage)(predictions)
        return predictions

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if isinstance(batch, tuple):
            batch = batch[0]
        elif isinstance(batch, list):
            # Todo: Understand why stack is needed
            batch = torch.stack(batch)
        return self(batch)

    def configure_optimizers(self) -> Union[Optimizer, Tuple[List[Optimizer], List[_LRScheduler]]]:
        optimizer = self.optimizer
        if not isinstance(self.optimizer, Optimizer):
            self.optimizer_kwargs["lr"] = self.learning_rate
            optimizer = optimizer(filter(lambda p: p.requires_grad, self.parameters()), **self.optimizer_kwargs)
        if self.scheduler:
            return [optimizer], [self._instantiate_scheduler(optimizer)]
        return optimizer

    def configure_finetune_callback(self) -> List[Callback]:
        return []

    @staticmethod
    def _resolve(
        old_preprocess: Optional[Preprocess],
        old_postprocess: Optional[Postprocess],
        old_serializer: Optional[Serializer],
        new_preprocess: Optional[Preprocess],
        new_postprocess: Optional[Postprocess],
        new_serializer: Optional[Serializer],
    ) -> Tuple[Optional[Preprocess], Optional[Postprocess], Optional[Serializer]]:
        """Resolves the correct :class:`~flash.core.data.process.Preprocess`, :class:`~flash.core.data.process.Postprocess`, and
        :class:`~flash.core.data.process.Serializer` to use, choosing ``new_*`` if it is not None or a base class
        (:class:`~flash.core.data.process.Preprocess`, :class:`~flash.core.data.process.Postprocess`, or
        :class:`~flash.core.data.process.Serializer`) and ``old_*`` otherwise.

        Args:
            old_preprocess: :class:`~flash.core.data.process.Preprocess` to be overridden.
            old_postprocess: :class:`~flash.core.data.process.Postprocess` to be overridden.
            old_serializer: :class:`~flash.core.data.process.Serializer` to be overridden.
            new_preprocess: :class:`~flash.core.data.process.Preprocess` to override with.
            new_postprocess: :class:`~flash.core.data.process.Postprocess` to override with.
            new_serializer: :class:`~flash.core.data.process.Serializer` to override with.

        Returns:
            The resolved :class:`~flash.core.data.process.Preprocess`, :class:`~flash.core.data.process.Postprocess`,
            and :class:`~flash.core.data.process.Serializer`.
        """
        preprocess = old_preprocess
        if new_preprocess is not None and type(new_preprocess) != Preprocess:
            preprocess = new_preprocess

        postprocess = old_postprocess
        if new_postprocess is not None and type(new_postprocess) != Postprocess:
            postprocess = new_postprocess

        serializer = old_serializer
        if new_serializer is not None and type(new_serializer) != Serializer:
            serializer = new_serializer

        return preprocess, postprocess, serializer

    @property
    def serializer(self) -> Optional[Serializer]:
        """The current :class:`.Serializer` associated with this model. If this property was set to a mapping
        (e.g. ``.serializer = {'output1': SerializerOne()}``) then this will be a :class:`.MappingSerializer`."""
        return self._serializer

    @serializer.setter
    def serializer(self, serializer: Union[Serializer, Mapping[str, Serializer]]):
        if isinstance(serializer, Mapping):
            serializer = SerializerMapping(serializer)
        self._serializer = serializer

    def build_data_pipeline(
        self,
        data_source: Optional[str] = None,
        data_pipeline: Optional[DataPipeline] = None,
    ) -> Optional[DataPipeline]:
        """Build a :class:`.DataPipeline` incorporating available
        :class:`~flash.core.data.process.Preprocess` and :class:`~flash.core.data.process.Postprocess`
        objects. These will be overridden in the following resolution order (lowest priority first):

        - Lightning ``Datamodule``, either attached to the :class:`.Trainer` or to the :class:`.Task`.
        - :class:`.Task` defaults given to :meth:`.Task.__init__`.
        - :class:`.Task` manual overrides by setting :py:attr:`~data_pipeline`.
        - :class:`.DataPipeline` passed to this method.

        Args:
            data_pipeline: Optional highest priority source of
                :class:`~flash.core.data.process.Preprocess` and :class:`~flash.core.data.process.Postprocess`.

        Returns:
            The fully resolved :class:`.DataPipeline`.
        """
        old_data_source, preprocess, postprocess, serializer = None, None, None, None

        # Datamodule
        if self.datamodule is not None and getattr(self.datamodule, 'data_pipeline', None) is not None:
            old_data_source = getattr(self.datamodule.data_pipeline, 'data_source', None)
            preprocess = getattr(self.datamodule.data_pipeline, '_preprocess_pipeline', None)
            postprocess = getattr(self.datamodule.data_pipeline, '_postprocess_pipeline', None)
            serializer = getattr(self.datamodule.data_pipeline, '_serializer', None)

        elif self.trainer is not None and hasattr(self.trainer, 'datamodule') and getattr(
            self.trainer.datamodule, 'data_pipeline', None
        ) is not None:
            old_data_source = getattr(self.trainer.datamodule.data_pipeline, 'data_source', None)
            preprocess = getattr(self.trainer.datamodule.data_pipeline, '_preprocess_pipeline', None)
            postprocess = getattr(self.trainer.datamodule.data_pipeline, '_postprocess_pipeline', None)
            serializer = getattr(self.trainer.datamodule.data_pipeline, '_serializer', None)
        else:
            # TODO: we should log with low severity level that we use defaults to create
            # `preprocess`, `postprocess` and `serializer`.
            pass

        # Defaults / task attributes
        preprocess, postprocess, serializer = Task._resolve(
            preprocess,
            postprocess,
            serializer,
            self._preprocess,
            self._postprocess,
            self.serializer,
        )

        # Datapipeline
        if data_pipeline is not None:
            preprocess, postprocess, serializer = Task._resolve(
                preprocess,
                postprocess,
                serializer,
                getattr(data_pipeline, '_preprocess_pipeline', None),
                getattr(data_pipeline, '_postprocess_pipeline', None),
                getattr(data_pipeline, '_serializer', None),
            )

        data_source = data_source or old_data_source

        if isinstance(data_source, str):
            if preprocess is None:
                data_source = DataSource()  # TODO: warn the user that we are not using the specified data source
            else:
                data_source = preprocess.data_source_of_name(data_source)

        data_pipeline = DataPipeline(data_source, preprocess, postprocess, serializer)
        self._data_pipeline_state = data_pipeline.initialize(self._data_pipeline_state)
        return data_pipeline

    @property
    def data_pipeline(self) -> DataPipeline:
        """The current :class:`.DataPipeline`. If set, the new value will override the :class:`.Task` defaults. See
        :py:meth:`~build_data_pipeline` for more details on the resolution order."""
        return self.build_data_pipeline()

    @data_pipeline.setter
    def data_pipeline(self, data_pipeline: Optional[DataPipeline]) -> None:
        self._preprocess, self._postprocess, self.serializer = Task._resolve(
            self._preprocess,
            self._postprocess,
            self.serializer,
            getattr(data_pipeline, '_preprocess_pipeline', None),
            getattr(data_pipeline, '_postprocess_pipeline', None),
            getattr(data_pipeline, '_serializer', None),
        )
        self._preprocess.state_dict()
        if getattr(self._preprocess, "_ddp_params_and_buffers_to_ignore", None):
            self._ddp_params_and_buffers_to_ignore = self._preprocess._ddp_params_and_buffers_to_ignore

    @property
    def preprocess(self) -> Preprocess:
        return getattr(self.data_pipeline, '_preprocess_pipeline', None)

    @property
    def postprocess(self) -> Postprocess:
        return getattr(self.data_pipeline, '_postprocess_pipeline', None)

    def on_train_dataloader(self) -> None:
        if self.data_pipeline is not None:
            self.data_pipeline._detach_from_model(self, RunningStage.TRAINING)
            self.data_pipeline._attach_to_model(self, RunningStage.TRAINING)
        super().on_train_dataloader()

    def on_val_dataloader(self) -> None:
        if self.data_pipeline is not None:
            self.data_pipeline._detach_from_model(self, RunningStage.VALIDATING)
            self.data_pipeline._attach_to_model(self, RunningStage.VALIDATING)
        super().on_val_dataloader()

    def on_test_dataloader(self, *_) -> None:
        if self.data_pipeline is not None:
            self.data_pipeline._detach_from_model(self, RunningStage.TESTING)
            self.data_pipeline._attach_to_model(self, RunningStage.TESTING)
        super().on_test_dataloader()

    def on_predict_dataloader(self) -> None:
        if self.data_pipeline is not None:
            self.data_pipeline._detach_from_model(self, RunningStage.PREDICTING)
            self.data_pipeline._attach_to_model(self, RunningStage.PREDICTING)
        super().on_predict_dataloader()

    def on_predict_end(self) -> None:
        if self.data_pipeline is not None:
            self.data_pipeline._detach_from_model(self)
        super().on_predict_end()

    def on_fit_end(self) -> None:
        if self.data_pipeline is not None:
            self.data_pipeline._detach_from_model(self)
        super().on_fit_end()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # This may be an issue since here we create the same problems with pickle as in
        # https://pytorch.org/docs/stable/notes/serialization.html
        if self.data_pipeline is not None and 'data_pipeline' not in checkpoint:
            checkpoint['data_pipeline'] = self.data_pipeline
        if self._data_pipeline_state is not None and '_data_pipeline_state' not in checkpoint:
            checkpoint['_data_pipeline_state'] = self._data_pipeline_state
        super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        if 'data_pipeline' in checkpoint:
            self.data_pipeline = checkpoint['data_pipeline']
        if '_data_pipeline_state' in checkpoint:
            self._data_pipeline_state = checkpoint['_data_pipeline_state']

    @classmethod
    def available_backbones(cls) -> List[str]:
        registry: Optional[FlashRegistry] = getattr(cls, "backbones", None)
        if registry is None:
            return []
        return registry.available_keys()

    @classmethod
    def available_models(cls) -> List[str]:
        registry: Optional[FlashRegistry] = getattr(cls, "models", None)
        if registry is None:
            return []
        return registry.available_keys()

    @classmethod
    def get_backbone_details(cls, key) -> List[str]:
        registry: Optional[FlashRegistry] = getattr(cls, "backbones", None)
        if registry is None:
            return []
        return [v for v in inspect.signature(registry.get(key)).parameters.items()]

    @classmethod
    def available_schedulers(cls) -> List[str]:
        registry: Optional[FlashRegistry] = getattr(cls, "schedulers", None)
        if registry is None:
            return []
        return registry.available_keys()

    def get_num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if not getattr(self, "trainer", None):
            raise MisconfigurationException("The LightningModule isn't attached to the trainer yet.")
        if isinstance(self.trainer.limit_train_batches, int) and self.trainer.limit_train_batches != 0:
            dataset_size = self.trainer.limit_train_batches
        elif isinstance(self.trainer.limit_train_batches, float):
            # limit_train_batches is a percentage of batches
            dataset_size = len(self.train_dataloader())
            dataset_size = int(dataset_size * self.trainer.limit_train_batches)
        else:
            dataset_size = len(self.train_dataloader())

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs

        if self.trainer.max_steps and self.trainer.max_steps < max_estimated_steps:
            return self.trainer.max_steps
        return max_estimated_steps

    def _compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> int:
        if not isinstance(num_warmup_steps, float) or (num_warmup_steps > 1 or num_warmup_steps < 0):
            raise MisconfigurationException(
                "`num_warmup_steps` should be provided as float between 0 and 1 in `scheduler_kwargs`"
            )
        if isinstance(num_warmup_steps, float):
            # Convert float values to percentage of training steps to use as warmup
            num_warmup_steps *= num_training_steps
        return round(num_warmup_steps)

    def _instantiate_scheduler(self, optimizer: Optimizer) -> _LRScheduler:
        scheduler = self.scheduler
        if isinstance(scheduler, _LRScheduler):
            return scheduler
        if isinstance(scheduler, str):
            scheduler_fn = self.schedulers.get(self.scheduler)
            num_training_steps: int = self.get_num_training_steps()
            num_warmup_steps: int = self._compute_warmup(
                num_training_steps=num_training_steps,
                num_warmup_steps=self.scheduler_kwargs.get("num_warmup_steps"),
            )
            return scheduler_fn(optimizer, num_warmup_steps, num_training_steps)
        elif issubclass(scheduler, _LRScheduler):
            return scheduler(optimizer, **self.scheduler_kwargs)
        raise MisconfigurationException(
            "scheduler can be a scheduler, a scheduler type with `scheduler_kwargs` "
            f"or a built-in scheduler in {self.available_schedulers()}"
        )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if 'preprocess.state_dict' in state_dict:
            try:
                preprocess_state_dict = state_dict["preprocess.state_dict"]
                meta = preprocess_state_dict["_meta"]
                cls = getattr(import_module(meta["module"]), meta["class_name"])
                self._preprocess = cls.load_state_dict(
                    {k: v
                     for k, v in preprocess_state_dict.items() if k != '_meta'},
                    strict=strict,
                )
                self._preprocess._state = meta["_state"]
                del state_dict["preprocess.state_dict"]
                del preprocess_state_dict["_meta"]
            except (ModuleNotFoundError, KeyError):
                meta = state_dict["preprocess.state_dict"]["_meta"]
                raise MisconfigurationException(
                    f"The `Preprocess` {meta['module']}.{meta['class_name']} has been moved and couldn't be imported."
                )

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def configure_callbacks(self):
        # used only for CI
        if flash._IS_TESTING and torch.cuda.is_available():
            return [BenchmarkConvergenceCI()]
