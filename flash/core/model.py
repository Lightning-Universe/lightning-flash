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
import pickle
from abc import ABCMeta
from copy import deepcopy
from importlib import import_module
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Sampler

import flash
from flash.core.data.auto_dataset import BaseAutoDataset
from flash.core.data.data_pipeline import DataPipeline, DataPipelineState
from flash.core.data.data_source import DataSource
from flash.core.data.process import (
    Deserializer,
    DeserializerMapping,
    Postprocess,
    Preprocess,
    Serializer,
    SerializerMapping,
)
from flash.core.data.properties import ProcessState
from flash.core.registry import FlashRegistry
from flash.core.schedulers import _SCHEDULERS_REGISTRY
from flash.core.serve import Composition
from flash.core.utilities.apply_func import get_callable_dict
from flash.core.utilities.imports import requires_extras


class ModuleWrapperBase:
    """The ``ModuleWrapperBase`` is a base for classes which wrap a ``LightningModule`` or an instance of
    ``ModuleWrapperBase``.

    This class ensures that trainer attributes are forwarded to any wrapped or nested
    ``LightningModule`` instances so that nested calls to ``.log`` are handled correctly. The ``ModuleWrapperBase`` is
    also stateful, meaning that a :class:`~flash.core.data.data_pipeline.DataPipelineState` can be attached. Attached
    state will be forwarded to any nested ``ModuleWrapperBase`` instances.
    """

    def __init__(self):
        super().__init__()

        self._children = []

        # TODO: create enum values to define what are the exact states
        self._data_pipeline_state: Optional[DataPipelineState] = None

        # model own internal state shared with the data pipeline.
        self._state: Dict[Type[ProcessState], ProcessState] = {}

    def __setattr__(self, key, value):
        if isinstance(value, (LightningModule, ModuleWrapperBase)):
            self._children.append(key)
        patched_attributes = ["_current_fx_name", "_current_hook_fx_name", "_results", "_data_pipeline_state"]
        if isinstance(value, Trainer) or key in patched_attributes:
            if hasattr(self, "_children"):
                for child in self._children:
                    setattr(getattr(self, child), key, value)
        super().__setattr__(key, value)

    def get_state(self, state_type):
        if state_type in self._state:
            return self._state[state_type]
        if self._data_pipeline_state is not None:
            return self._data_pipeline_state.get_state(state_type)
        return None

    def set_state(self, state: ProcessState):
        self._state[type(state)] = state
        if self._data_pipeline_state is not None:
            self._data_pipeline_state.set_state(state)

    def attach_data_pipeline_state(self, data_pipeline_state: "DataPipelineState"):
        for state in self._state.values():
            data_pipeline_state.set_state(state)
        for child in self._children:
            child = getattr(self, child)
            if hasattr(child, "attach_data_pipeline_state"):
                child.attach_data_pipeline_state(data_pipeline_state)


class DatasetProcessor:
    """The ``DatasetProcessor`` mixin provides hooks for classes which need custom logic for producing the data
    loaders for each running stage given the corresponding dataset."""

    def _process_dataset(
        self,
        dataset: BaseAutoDataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        collate_fn: Callable,
        shuffle: bool = False,
        drop_last: bool = True,
        sampler: Optional[Sampler] = None,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
            collate_fn=collate_fn,
        )

    def process_train_dataset(
        self,
        dataset: BaseAutoDataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        collate_fn: Callable,
        shuffle: bool = False,
        drop_last: bool = True,
        sampler: Optional[Sampler] = None,
    ) -> DataLoader:
        return self._process_dataset(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
        )

    def process_val_dataset(
        self,
        dataset: BaseAutoDataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        collate_fn: Callable,
        shuffle: bool = False,
        drop_last: bool = False,
        sampler: Optional[Sampler] = None,
    ) -> DataLoader:
        return self._process_dataset(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
        )

    def process_test_dataset(
        self,
        dataset: BaseAutoDataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        collate_fn: Callable,
        shuffle: bool = False,
        drop_last: bool = False,
        sampler: Optional[Sampler] = None,
    ) -> DataLoader:
        return self._process_dataset(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
        )

    def process_predict_dataset(
        self,
        dataset: BaseAutoDataset,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        collate_fn: Callable = None,
        shuffle: bool = False,
        drop_last: bool = False,
        sampler: Optional[Sampler] = None,
    ) -> DataLoader:
        return self._process_dataset(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
        )


class BenchmarkConvergenceCI(Callback):
    def __init__(self):
        self.history = []

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.history.append(deepcopy(trainer.callback_metrics))
        if trainer.current_epoch == trainer.max_epochs - 1:
            fn = getattr(pl_module, "_ci_benchmark_fn", None)
            if fn:
                fn(self.history)
                if trainer.is_global_zero:
                    print("Benchmark Successful!")


def predict_context(func: Callable) -> Callable:
    """This decorator is used as context manager to put model in eval mode before running predict and reset to
    train after."""

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


class CheckDependenciesMeta(ABCMeta):
    def __new__(mcs, *args, **kwargs):
        result = ABCMeta.__new__(mcs, *args, **kwargs)
        if result.required_extras is not None:
            result.__init__ = requires_extras(result.required_extras)(result.__init__)
            load_from_checkpoint = getattr(result, "load_from_checkpoint", None)
            if load_from_checkpoint is not None:
                result.load_from_checkpoint = classmethod(
                    requires_extras(result.required_extras)(result.load_from_checkpoint.__func__)
                )
        return result


class Task(DatasetProcessor, ModuleWrapperBase, LightningModule, metaclass=CheckDependenciesMeta):
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

    required_extras: Optional[str] = None

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
        deserializer: Optional[Union[Deserializer, Mapping[str, Deserializer]]] = None,
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

        self.train_metrics = nn.ModuleDict({} if metrics is None else get_callable_dict(metrics))
        self.val_metrics = nn.ModuleDict({} if metrics is None else get_callable_dict(deepcopy(metrics)))
        self.learning_rate = learning_rate
        # TODO: should we save more? Bug on some regarding yaml if we save metrics
        self.save_hyperparameters("learning_rate", "optimizer")

        self._deserializer: Optional[Deserializer] = None
        self._preprocess: Optional[Preprocess] = preprocess
        self._postprocess: Optional[Postprocess] = postprocess
        self._serializer: Optional[Serializer] = None

        # Explicitly set the serializer to call the setter
        self.deserializer = deserializer
        self.serializer = serializer

    def step(self, batch: Any, batch_idx: int, metrics: nn.ModuleDict) -> Any:
        """The training/validation/test step.

        Override for custom behavior.
        """
        x, y = batch
        y_hat = self(x)
        y, y_hat = self.apply_filtering(y, y_hat)
        output = {"y_hat": y_hat}
        y_hat = self.to_loss_format(output["y_hat"])
        losses = {name: l_fn(y_hat, y) for name, l_fn in self.loss_fn.items()}

        y_hat = self.to_metrics_format(output["y_hat"])

        logs = {}

        for name, metric in metrics.items():
            if isinstance(metric, torchmetrics.metric.Metric):
                metric(y_hat, y)
                logs[name] = metric  # log the metric itself if it is of type Metric
            else:
                logs[name] = metric(y_hat, y)

        if len(losses.values()) > 1:
            logs["total_loss"] = sum(losses.values())
            return logs["total_loss"], logs

        output["loss"] = self.compute_loss(losses)
        output["logs"] = self.compute_logs(logs, losses)
        output["y"] = y
        return output

    def compute_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        return list(losses.values())[0]

    def compute_logs(self, logs: Dict[str, Any], losses: Dict[str, torch.Tensor]):
        logs.update(losses)
        return logs

    @staticmethod
    def apply_filtering(y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """This function is used to filter some labels or predictions which aren't conform."""
        return y, y_hat

    @staticmethod
    def to_loss_format(x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def to_metrics_format(x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        output = self.step(batch, batch_idx, self.train_metrics)
        self.log_dict({f"train_{k}": v for k, v in output["logs"].items()}, on_step=True, on_epoch=True, prog_bar=True)
        return output["loss"]

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        output = self.step(batch, batch_idx, self.val_metrics)
        self.log_dict({f"val_{k}": v for k, v in output["logs"].items()}, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        output = self.step(batch, batch_idx, self.val_metrics)
        self.log_dict({f"test_{k}": v for k, v in output["logs"].items()}, on_step=False, on_epoch=True, prog_bar=True)

    @predict_context
    def predict(
        self,
        x: Any,
        data_source: Optional[str] = None,
        deserializer: Optional[Deserializer] = None,
        data_pipeline: Optional[DataPipeline] = None,
    ) -> Any:
        """Predict function for raw data or processed data.

        Args:
            x: Input to predict. Can be raw data or processed data. If str, assumed to be a folder of data.

            data_pipeline: Use this to override the current data pipeline

        Returns:
            The post-processed model predictions
        """
        running_stage = RunningStage.PREDICTING

        data_pipeline = self.build_data_pipeline(data_source or "default", deserializer, data_pipeline)
        dataset = data_pipeline.data_source.generate_dataset(x, running_stage)
        dataloader = self.process_predict_dataset(dataset)
        x = list(dataloader.dataset)
        x = data_pipeline.worker_preprocessor(running_stage, collate_fn=dataloader.collate_fn)(x)
        # todo (tchaton): Remove this when sync with Lightning master.
        if len(inspect.signature(self.transfer_batch_to_device).parameters) == 3:
            x = self.transfer_batch_to_device(x, self.device, 0)
        else:
            x = self.transfer_batch_to_device(x, self.device)
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

    @staticmethod
    def configure_finetune_callback() -> List[Callback]:
        return []

    @staticmethod
    def _resolve(
        old_deserializer: Optional[Deserializer],
        old_preprocess: Optional[Preprocess],
        old_postprocess: Optional[Postprocess],
        old_serializer: Optional[Serializer],
        new_deserializer: Optional[Deserializer],
        new_preprocess: Optional[Preprocess],
        new_postprocess: Optional[Postprocess],
        new_serializer: Optional[Serializer],
    ) -> Tuple[Optional[Deserializer], Optional[Preprocess], Optional[Postprocess], Optional[Serializer]]:
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
        deserializer = old_deserializer
        if new_deserializer is not None and type(new_deserializer) != Deserializer:
            deserializer = new_deserializer

        preprocess = old_preprocess
        if new_preprocess is not None and type(new_preprocess) != Preprocess:
            preprocess = new_preprocess

        postprocess = old_postprocess
        if new_postprocess is not None and type(new_postprocess) != Postprocess:
            postprocess = new_postprocess

        serializer = old_serializer
        if new_serializer is not None and type(new_serializer) != Serializer:
            serializer = new_serializer

        return deserializer, preprocess, postprocess, serializer

    @torch.jit.unused
    @property
    def deserializer(self) -> Optional[Deserializer]:
        return self._deserializer

    @deserializer.setter
    def deserializer(self, deserializer: Union[Deserializer, Mapping[str, Deserializer]]):
        if isinstance(deserializer, Mapping):
            deserializer = DeserializerMapping(deserializer)
        self._deserializer = deserializer

    @torch.jit.unused
    @property
    def serializer(self) -> Optional[Serializer]:
        """The current :class:`.Serializer` associated with this model.

        If this property was set to a mapping
        (e.g. ``.serializer = {'output1': SerializerOne()}``) then this will be a :class:`.MappingSerializer`.
        """
        return self._serializer

    @torch.jit.unused
    @serializer.setter
    def serializer(self, serializer: Union[Serializer, Mapping[str, Serializer]]):
        if isinstance(serializer, Mapping):
            serializer = SerializerMapping(serializer)
        self._serializer = serializer

    def build_data_pipeline(
        self,
        data_source: Optional[str] = None,
        deserializer: Optional[Deserializer] = None,
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
        deserializer, old_data_source, preprocess, postprocess, serializer = None, None, None, None, None

        # Datamodule
        if self.datamodule is not None and getattr(self.datamodule, "data_pipeline", None) is not None:
            old_data_source = getattr(self.datamodule.data_pipeline, "data_source", None)
            preprocess = getattr(self.datamodule.data_pipeline, "_preprocess_pipeline", None)
            postprocess = getattr(self.datamodule.data_pipeline, "_postprocess_pipeline", None)
            serializer = getattr(self.datamodule.data_pipeline, "_serializer", None)
            deserializer = getattr(self.datamodule.data_pipeline, "_deserializer", None)

        elif (
            self.trainer is not None
            and hasattr(self.trainer, "datamodule")
            and getattr(self.trainer.datamodule, "data_pipeline", None) is not None
        ):
            old_data_source = getattr(self.trainer.datamodule.data_pipeline, "data_source", None)
            preprocess = getattr(self.trainer.datamodule.data_pipeline, "_preprocess_pipeline", None)
            postprocess = getattr(self.trainer.datamodule.data_pipeline, "_postprocess_pipeline", None)
            serializer = getattr(self.trainer.datamodule.data_pipeline, "_serializer", None)
            deserializer = getattr(self.trainer.datamodule.data_pipeline, "_deserializer", None)
        else:
            # TODO: we should log with low severity level that we use defaults to create
            # `preprocess`, `postprocess` and `serializer`.
            pass

        # Defaults / task attributes
        deserializer, preprocess, postprocess, serializer = Task._resolve(
            deserializer,
            preprocess,
            postprocess,
            serializer,
            self._deserializer,
            self._preprocess,
            self._postprocess,
            self._serializer,
        )

        # Datapipeline
        if data_pipeline is not None:
            deserializer, preprocess, postprocess, serializer = Task._resolve(
                deserializer,
                preprocess,
                postprocess,
                serializer,
                getattr(data_pipeline, "_deserializer", None),
                getattr(data_pipeline, "_preprocess_pipeline", None),
                getattr(data_pipeline, "_postprocess_pipeline", None),
                getattr(data_pipeline, "_serializer", None),
            )

        data_source = data_source or old_data_source

        if isinstance(data_source, str):
            if preprocess is None:
                data_source = DataSource()  # TODO: warn the user that we are not using the specified data source
            else:
                data_source = preprocess.data_source_of_name(data_source)

        if deserializer is None or type(deserializer) is Deserializer:
            deserializer = getattr(preprocess, "deserializer", deserializer)

        data_pipeline = DataPipeline(data_source, preprocess, postprocess, deserializer, serializer)
        self._data_pipeline_state = self._data_pipeline_state or DataPipelineState()
        self.attach_data_pipeline_state(self._data_pipeline_state)
        self._data_pipeline_state = data_pipeline.initialize(self._data_pipeline_state)
        return data_pipeline

    @torch.jit.unused
    @property
    def is_servable(self) -> bool:
        return type(self.build_data_pipeline()._deserializer) != Deserializer

    @torch.jit.unused
    @property
    def data_pipeline(self) -> DataPipeline:
        """The current :class:`.DataPipeline`.

        If set, the new value will override the :class:`.Task` defaults. See
        :py:meth:`~build_data_pipeline` for more details on the resolution order.
        """
        return self.build_data_pipeline()

    @torch.jit.unused
    @data_pipeline.setter
    def data_pipeline(self, data_pipeline: Optional[DataPipeline]) -> None:
        self._deserializer, self._preprocess, self._postprocess, self.serializer = Task._resolve(
            self._deserializer,
            self._preprocess,
            self._postprocess,
            self._serializer,
            getattr(data_pipeline, "_deserializer", None),
            getattr(data_pipeline, "_preprocess_pipeline", None),
            getattr(data_pipeline, "_postprocess_pipeline", None),
            getattr(data_pipeline, "_serializer", None),
        )

        # self._preprocess.state_dict()
        if getattr(self._preprocess, "_ddp_params_and_buffers_to_ignore", None):
            self._ddp_params_and_buffers_to_ignore = self._preprocess._ddp_params_and_buffers_to_ignore

    @torch.jit.unused
    @property
    def preprocess(self) -> Preprocess:
        return getattr(self.data_pipeline, "_preprocess_pipeline", None)

    @torch.jit.unused
    @property
    def postprocess(self) -> Postprocess:
        return getattr(self.data_pipeline, "_postprocess_pipeline", None)

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
        if self.data_pipeline is not None and "data_pipeline" not in checkpoint:
            try:
                pickle.dumps(self.data_pipeline)  # TODO: DataPipeline not always pickleable
                checkpoint["data_pipeline"] = self.data_pipeline
            except AttributeError:
                rank_zero_warn("DataPipeline couldn't be added to the checkpoint.")
        if self._data_pipeline_state is not None and "_data_pipeline_state" not in checkpoint:
            checkpoint["_data_pipeline_state"] = self._data_pipeline_state
        super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        if "data_pipeline" in checkpoint:
            self.data_pipeline = checkpoint["data_pipeline"]
        if "_data_pipeline_state" in checkpoint:
            self._data_pipeline_state = checkpoint["_data_pipeline_state"]

    @classmethod
    def available_backbones(cls, head: Optional[str] = None) -> Union[Dict[str, List[str]], List[str]]:
        if head is None:
            registry: Optional[FlashRegistry] = getattr(cls, "backbones", None)
            if registry is not None:
                return registry.available_keys()
            heads = cls.available_heads()
        else:
            heads = [head]

        result = {}
        for head in heads:
            metadata = cls.heads.get(head, with_metadata=True)["metadata"]
            if "backbones" in metadata:
                backbones = metadata["backbones"].available_keys()
            else:
                backbones = cls.available_backbones()
            result[head] = backbones

        if len(result) == 1:
            result = next(iter(result.values()))
        return result

    @classmethod
    def available_heads(cls) -> List[str]:
        registry: Optional[FlashRegistry] = getattr(cls, "heads", None)
        if registry is None:
            return []
        return registry.available_keys()

    @classmethod
    def get_backbone_details(cls, key) -> List[str]:
        registry: Optional[FlashRegistry] = getattr(cls, "backbones", None)
        if registry is None:
            return []
        return list(inspect.signature(registry.get(key)).parameters.items())

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

    @staticmethod
    def _compute_warmup(num_training_steps: int, num_warmup_steps: Union[int, float]) -> int:
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
        if issubclass(scheduler, _LRScheduler):
            return scheduler(optimizer, **self.scheduler_kwargs)
        raise MisconfigurationException(
            "scheduler can be a scheduler, a scheduler type with `scheduler_kwargs` "
            f"or a built-in scheduler in {self.available_schedulers()}"
        )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if "preprocess.state_dict" in state_dict:
            try:
                preprocess_state_dict = state_dict["preprocess.state_dict"]
                meta = preprocess_state_dict["_meta"]
                cls = getattr(import_module(meta["module"]), meta["class_name"])
                self._preprocess = cls.load_state_dict(
                    {k: v for k, v in preprocess_state_dict.items() if k != "_meta"},
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

    @requires_extras("serve")
    def run_serve_sanity_check(self):
        if not self.is_servable:
            raise NotImplementedError("This Task is not servable. Attach a Deserializer to enable serving.")

        from fastapi.testclient import TestClient

        from flash.core.serve.flash_components import build_flash_serve_model_component

        print("Running serve sanity check")
        comp = build_flash_serve_model_component(self)
        composition = Composition(predict=comp, TESTING=True, DEBUG=True)
        app = composition.serve(host="0.0.0.0", port=8000)

        with TestClient(app) as tc:
            input_str = self.data_pipeline._deserializer.example_input
            body = {"session": "UUID", "payload": {"inputs": {"data": input_str}}}
            resp = tc.post("http://0.0.0.0:8000/predict", json=body)
            print(f"Sanity check response: {resp.json()}")

    @requires_extras("serve")
    def serve(self, host: str = "127.0.0.1", port: int = 8000, sanity_check: bool = True) -> "Composition":
        if not self.is_servable:
            raise NotImplementedError("This Task is not servable. Attach a Deserializer to enable serving.")

        from flash.core.serve.flash_components import build_flash_serve_model_component

        if sanity_check:
            self.run_serve_sanity_check()

        comp = build_flash_serve_model_component(self)
        composition = Composition(predict=comp, TESTING=flash._IS_TESTING)
        composition.serve(host=host, port=port)
        return composition
