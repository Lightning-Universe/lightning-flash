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
import re
from abc import ABCMeta
from copy import deepcopy
from importlib import import_module
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, Union
from warnings import warn

import pytorch_lightning as pl
import torch
import torchmetrics
from deprecate import deprecated
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.enums import LightningEnum
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Sampler

import flash
from flash.core.data.data_pipeline import DataPipeline, DataPipelineState
from flash.core.data.io.input import InputBase, ServeInput
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.io.output import Output
from flash.core.data.io.output_transform import OutputTransform
from flash.core.data.process import Deserializer, DeserializerMapping
from flash.core.data.properties import ProcessState
from flash.core.finetuning import _DEFAULTS_FINETUNE_STRATEGIES, _FINETUNING_STRATEGIES_REGISTRY
from flash.core.hooks import FineTuningHooks
from flash.core.optimizers.optimizers import _OPTIMIZERS_REGISTRY
from flash.core.optimizers.schedulers import _SCHEDULERS_REGISTRY
from flash.core.registry import FlashRegistry
from flash.core.serve.composition import Composition
from flash.core.utilities.apply_func import get_callable_dict
from flash.core.utilities.imports import _PL_GREATER_EQUAL_1_5_0, requires
from flash.core.utilities.providers import _HUGGINGFACE
from flash.core.utilities.stages import RunningStage
from flash.core.utilities.types import (
    DESERIALIZER_TYPE,
    INPUT_TRANSFORM_TYPE,
    LOSS_FN_TYPE,
    LR_SCHEDULER_TYPE,
    METRICS_TYPE,
    MODEL_TYPE,
    OPTIMIZER_TYPE,
    OUTPUT_TRANSFORM_TYPE,
    OUTPUT_TYPE,
)


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
        self._data_pipeline_state: DataPipelineState = DataPipelineState()

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
        if self._data_pipeline_state:
            for state in self._data_pipeline_state._state.values():
                data_pipeline_state.set_state(state)
        self._data_pipeline_state = data_pipeline_state
        for child in self._children:
            child = getattr(self, child)
            if hasattr(child, "attach_data_pipeline_state"):
                child.attach_data_pipeline_state(data_pipeline_state)


class DatasetProcessor:
    """The ``DatasetProcessor`` mixin provides hooks for classes which need custom logic for producing the data
    loaders for each running stage given the corresponding dataset."""

    def _process_dataset(
        self,
        dataset: InputBase,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        collate_fn: Callable,
        shuffle: bool = False,
        drop_last: bool = True,
        sampler: Optional[Sampler] = None,
        persistent_workers: bool = True,
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
            persistent_workers=persistent_workers,
        )

    def process_train_dataset(
        self,
        dataset: InputBase,
        trainer: "flash.Trainer",
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
            persistent_workers=num_workers > 0,
        )

    def process_val_dataset(
        self,
        dataset: InputBase,
        trainer: "flash.Trainer",
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
            persistent_workers=num_workers > 0,
        )

    def process_test_dataset(
        self,
        dataset: InputBase,
        trainer: "flash.Trainer",
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
            persistent_workers=num_workers > 0,
        )

    def process_predict_dataset(
        self,
        dataset: InputBase,
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
            persistent_workers=False,
        )


class BenchmarkConvergenceCI(Callback):
    """Specialized callback only used during testing Keeps track metrics during training."""

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


class CheckDependenciesMeta(ABCMeta):
    def __new__(mcs, *args, **kwargs):
        result = ABCMeta.__new__(mcs, *args, **kwargs)
        if result.required_extras is not None:
            result.__init__ = requires(result.required_extras)(result.__init__)

            patterns = ["load_from_checkpoint", "available_*"]  # must match classmethods only
            regex = "(" + ")|(".join(patterns) + ")"
            for attribute_name, attribute_value in filter(lambda x: re.match(regex, x[0]), inspect.getmembers(result)):
                setattr(result, attribute_name, classmethod(requires(result.required_extras)(attribute_value.__func__)))
        return result


class OutputKeys(LightningEnum):
    """The ``OutputKeys`` enum contains the keys that are used internally by the ``Task`` when handling outputs."""

    OUTPUT = "y_hat"
    TARGET = "y"
    LOGS = "logs"
    LOSS = "loss"

    # TODO: Create a FlashEnum class???
    def __hash__(self) -> int:
        return hash(self.value)


class Task(DatasetProcessor, ModuleWrapperBase, LightningModule, FineTuningHooks, metaclass=CheckDependenciesMeta):
    """A general Task.

    Args:
        model: Model to use for the task.
        loss_fn: Loss function for training.
        learning_rate: Learning rate to use for training, defaults to ``5e-5``.
        optimizer: Optimizer to use for training.
        lr_scheduler: The LR scheduler to use during training.
        metrics: Metrics to compute for training and evaluation. Can either be an metric from the `torchmetrics`
            package, a custom metric inheriting from `torchmetrics.Metric`, a callable function or a list/dict
            containing a combination of the aforementioned. In all cases, each metric needs to have the signature
            `metric(preds,target)` and return a single scalar tensor.
        deserializer: Either a single :class:`~flash.core.data.process.Deserializer` or a mapping of these to
            deserialize the input
        input_transform: :class:`~flash.core.data.io.input_transform.InputTransform` to use as the default
            for this task.
        output_transform: :class:`~flash.core.data.io.output_transform.OutputTransform` to use as the default for this
            task.
        output: The :class:`~flash.core.data.io.output.Output` to use when formatting prediction outputs.
    """

    optimizers: FlashRegistry = _OPTIMIZERS_REGISTRY
    lr_schedulers: FlashRegistry = _SCHEDULERS_REGISTRY
    finetuning_strategies: FlashRegistry = _FINETUNING_STRATEGIES_REGISTRY

    required_extras: Optional[Union[str, List[str]]] = None

    def __init__(
        self,
        model: MODEL_TYPE = None,
        loss_fn: LOSS_FN_TYPE = None,
        learning_rate: float = 5e-5,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        metrics: METRICS_TYPE = None,
        deserializer: DESERIALIZER_TYPE = None,
        input_transform: INPUT_TRANSFORM_TYPE = None,
        output_transform: OUTPUT_TRANSFORM_TYPE = None,
        output: OUTPUT_TYPE = None,
    ):
        super().__init__()
        if model is not None:
            self.model = model
        self.loss_fn = {} if loss_fn is None else get_callable_dict(loss_fn)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.train_metrics = nn.ModuleDict({} if metrics is None else get_callable_dict(metrics))
        self.val_metrics = nn.ModuleDict({} if metrics is None else get_callable_dict(deepcopy(metrics)))
        self.test_metrics = nn.ModuleDict({} if metrics is None else get_callable_dict(deepcopy(metrics)))
        self.learning_rate = learning_rate
        # TODO: should we save more? Bug on some regarding yaml if we save metrics
        self.save_hyperparameters("learning_rate", "optimizer")

        self._deserializer: Optional[Deserializer] = None
        self._input_transform: Optional[InputTransform] = input_transform
        self._output_transform: Optional[OutputTransform] = output_transform
        self._output: Optional[Output] = None

        # Explicitly set the output to call the setter
        self.deserializer = deserializer
        self.output = output
        self._wrapped_predict_step = False

    def _wrap_predict_step(self) -> None:
        if not self._wrapped_predict_step:
            process_fn = self.build_data_pipeline().output_transform_processor(RunningStage.PREDICTING)

            predict_step = self.predict_step

            @functools.wraps(predict_step)
            def wrapper(*args, **kwargs):
                predictions = predict_step(*args, **kwargs)
                return process_fn(predictions)

            self._original_predict_step = self.predict_step
            self.predict_step = wrapper

            self._wrapped_predict_step = True

    def _unwrap_predict_step(self) -> None:
        if self._wrapped_predict_step:
            self.predict_step = self._original_predict_step
            del self._original_predict_step
            self._wrapped_predict_step = False

    def step(self, batch: Any, batch_idx: int, metrics: nn.ModuleDict) -> Any:
        """Implement the core logic for the training/validation/test step. By default this includes:

            - Inference on the current batch
            - Calculating the loss
            - Calculating relevant metrics

        Override for custom behavior.

        Args:
            batch: The output of your dataloader. Can either be a single Tensor or a list of Tensors.
            batch_idx: Integer displaying index of this batch
            metrics: A module dict containing metrics for calculating relevant training statitics

        Returns:
            A dict containing both the loss and relevant metrics
        """
        x, y = batch
        y_hat = self(x)
        y, y_hat = self.apply_filtering(y, y_hat)
        output = {OutputKeys.OUTPUT: y_hat}
        y_hat = self.to_loss_format(output[OutputKeys.OUTPUT])
        losses = {name: l_fn(y_hat, y) for name, l_fn in self.loss_fn.items()}

        y_hat = self.to_metrics_format(output[OutputKeys.OUTPUT])

        logs = {}

        for name, metric in metrics.items():
            if isinstance(metric, torchmetrics.metric.Metric):
                metric(y_hat, y)
                # PL 1.4.0 -> 1.4.9 tries to deepcopy the metric.
                # Sometimes _forward_cache is not a leaf, so we convert it to one.
                if not metric._forward_cache.is_leaf and not _PL_GREATER_EQUAL_1_5_0:
                    metric._forward_cache = metric._forward_cache.clone().detach()
                logs[name] = metric  # log the metric itself if it is of type Metric
            else:
                logs[name] = metric(y_hat, y)

        if len(losses.values()) > 1:
            logs["total_loss"] = sum(losses.values())
            return logs["total_loss"], logs

        output[OutputKeys.LOSS] = self.compute_loss(losses)
        output[OutputKeys.LOGS] = self.compute_logs(logs, losses)
        output[OutputKeys.TARGET] = y
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
        self.log_dict(
            {f"train_{k}": v for k, v in output[OutputKeys.LOGS].items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return output[OutputKeys.LOSS]

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        output = self.step(batch, batch_idx, self.val_metrics)
        self.log_dict(
            {f"val_{k}": v for k, v in output[OutputKeys.LOGS].items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch: Any, batch_idx: int) -> None:
        output = self.step(batch, batch_idx, self.test_metrics)
        self.log_dict(
            {f"test_{k}": v for k, v in output[OutputKeys.LOGS].items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def predict(self, *args, **kwargs):
        raise AttributeError("`flash.Task.predict` has been removed. Use `flash.Trainer.predict` instead.")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if isinstance(batch, tuple):
            batch = batch[0]
        elif isinstance(batch, list):
            # Todo: Understand why stack is needed
            batch = torch.stack(batch)
        return self(batch)

    def modules_to_freeze(self) -> Optional[Union[nn.Module]]:
        """By default, we try to get the ``backbone`` attribute from the task and return it or ``None`` if not
        present.

        Returns:
            The backbone ``Module`` to freeze or ``None`` if this task does not have a ``backbone`` attribute.
        """
        return getattr(self, "backbone", None)

    def _get_optimizer_class_from_registry(self, optimizer_key: str) -> Optimizer:
        if optimizer_key.lower() not in self.available_optimizers():
            raise KeyError(
                f"Please provide a valid optimizer name and make sure it is registerd with the Optimizer registry."
                f"\nUse `{self.__class__.__name__}.available_optimizers()` to list the available optimizers."
                f"\nList of available Optimizers: {self.available_optimizers()}."
            )
        optimizer_fn = self.optimizers.get(optimizer_key.lower())
        return optimizer_fn

    def configure_optimizers(self) -> Union[Optimizer, Tuple[List[Optimizer], List[_LRScheduler]]]:
        """Implement how optimizer and optionally learning rate schedulers should be configured."""
        if isinstance(self.optimizer, str):
            optimizer_fn = self._get_optimizer_class_from_registry(self.optimizer.lower())
            optimizers_kwargs: Dict[str, Any] = {"lr": self.learning_rate}
        elif isinstance(self.optimizer, Callable):
            optimizer_fn = self.optimizer
            optimizers_kwargs: Dict[str, Any] = {"lr": self.learning_rate}
        elif isinstance(self.optimizer, Tuple):
            if len(self.optimizer) != 2:
                raise MisconfigurationException(
                    f"The tuple configuration of an optimizer input must be of length 2 with the first index"
                    f" containing a str from {self.available_optimizers()} and the second index containing the"
                    f" required keyword arguments to initialize the Optimizer."
                )

            if not isinstance(self.optimizer[0], str):
                raise TypeError(
                    f"The first value in optimizer argument tuple should be a string but got {type(self.optimizer[0])}."
                )

            if not isinstance(self.optimizer[1], Dict):
                raise TypeError(
                    f"The second value in optimizer argument tuple should be of dict type but got "
                    f"{type(self.optimizer[1])}."
                )

            optimizer_fn: Callable = self._get_optimizer_class_from_registry(self.optimizer[0])
            optimizers_kwargs: Dict[str, Any] = self.optimizer[1]
            optimizers_kwargs["lr"] = self.learning_rate
        else:
            raise TypeError(
                f"""Optimizer should be of type string or callable or tuple(string, dictionary)
                but got {type(self.optimizer)}."""
            )

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        optimizer: Optimizer = optimizer_fn(model_parameters, **optimizers_kwargs)
        if self.lr_scheduler is not None:
            return [optimizer], [self._instantiate_lr_scheduler(optimizer)]
        return optimizer

    def configure_finetune_callback(
        self,
        strategy: Union[str, Tuple[str, int], Tuple[str, Tuple[Tuple[int, int], int]], BaseFinetuning] = "no_freeze",
        train_bn: bool = True,
    ) -> List[BaseFinetuning]:

        if isinstance(strategy, BaseFinetuning):
            return [strategy]

        if isinstance(strategy, str):
            if strategy not in self.available_finetuning_strategies():
                raise MisconfigurationException(
                    f"Please provide a valid strategy from {_DEFAULTS_FINETUNE_STRATEGIES[:3]}."
                )
            finetuning_strategy_fn: Callable = self.finetuning_strategies.get(key=strategy)
            finetuning_strategy_metadata = {"strategy_metadata": None, "train_bn": train_bn}
        elif isinstance(strategy, Tuple):
            if not isinstance(strategy[0], str) or strategy[0] not in self.available_finetuning_strategies():
                raise MisconfigurationException(
                    f"First input of `strategy` in a tuple configuration should be a string within"
                    f" {_DEFAULTS_FINETUNE_STRATEGIES[3:]}"
                )
            finetuning_strategy_fn: Callable = self.finetuning_strategies.get(key=strategy[0])
            finetuning_strategy_metadata = {"strategy_metadata": strategy[1], "train_bn": train_bn}
        else:
            raise MisconfigurationException(
                "`strategy` should be a ``pytorch_lightning.callbacks.BaseFinetuning``"
                f"callback or a str within {list(_DEFAULTS_FINETUNE_STRATEGIES[:3])}"
                f"or a tuple configuration with {list(_DEFAULTS_FINETUNE_STRATEGIES[3:])}"
            )

        return [finetuning_strategy_fn(**finetuning_strategy_metadata)]

    @staticmethod
    def _resolve(
        old_deserializer: Optional[Deserializer],
        old_input_transform: Optional[InputTransform],
        old_output_transform: Optional[OutputTransform],
        old_output: Optional[Output],
        new_deserializer: Optional[Deserializer],
        new_input_transform: Optional[InputTransform],
        new_output_transform: Optional[OutputTransform],
        new_output: Optional[Output],
    ) -> Tuple[Optional[Deserializer], Optional[InputTransform], Optional[OutputTransform], Optional[Output]]:
        """Resolves the correct :class:`~flash.core.data.io.input_transform.InputTransform`,
        :class:`~flash.core.data.io.output_transform.OutputTransform`, and :class:`~flash.core.data.io.output.Output`
        to use, choosing ``new_*`` if it is not None or a base class
        (:class:`~flash.core.data.io.input_transform.InputTransform`,
        :class:`~flash.core.data.io.output_transform.OutputTransform`, or :class:`~flash.core.data.io.output.Output`)
        and ``old_*`` otherwise.

        Args:
            old_input_transform: :class:`~flash.core.data.io.input_transform.InputTransform` to be overridden.
            old_output_transform: :class:`~flash.core.data.io.output_transform.OutputTransform` to be overridden.
            old_output: :class:`~flash.core.data.io.output.Output` to be overridden.
            new_input_transform: :class:`~flash.core.data.io.input_transform.InputTransform` to override with.
            new_output_transform: :class:`~flash.core.data.io.output_transform.OutputTransform` to override with.
            new_output: :class:`~flash.core.data.io.output.Output` to override with.

        Returns:
            The resolved :class:`~flash.core.data.io.input_transform.InputTransform`,
            :class:`~flash.core.data.io.output_transform.OutputTransform`, and
            :class:`~flash.core.data.io.output.Output`.
        """
        deserializer = old_deserializer
        if new_deserializer is not None and type(new_deserializer) != Deserializer:
            deserializer = new_deserializer

        input_transform = old_input_transform
        if new_input_transform is not None and type(new_input_transform) != InputTransform:
            input_transform = new_input_transform

        output_transform = old_output_transform
        if new_output_transform is not None and type(new_output_transform) != OutputTransform:
            output_transform = new_output_transform

        output = old_output
        if new_output is not None and type(new_output) != Output:
            output = new_output

        return deserializer, input_transform, output_transform, output

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
    def output(self) -> Optional[Output]:
        """The current :class:`.Output` associated with this model."""
        return self._output

    @torch.jit.unused
    @output.setter
    def output(self, output: Output):
        self._output = output

    @torch.jit.unused
    @property
    @deprecated(
        None,
        "0.6.0",
        "0.7.0",
        template_mgs="`Task.serializer` was deprecated in v%(deprecated_in)s in favor of `Task.output`. "
        "It will be removed in v%(remove_in)s.",
        stream=functools.partial(warn, category=FutureWarning),
    )
    def serializer(self) -> Optional[Output]:
        """Deprecated.

        Use ``Task.output`` instead.
        """
        return self.output

    @torch.jit.unused
    @serializer.setter
    @deprecated(
        None,
        "0.6.0",
        "0.7.0",
        template_mgs="`Task.serializer` was deprecated in v%(deprecated_in)s in favor of `Task.output`. "
        "It will be removed in v%(remove_in)s.",
        stream=functools.partial(warn, category=FutureWarning),
    )
    def serializer(self, serializer: Output):
        self.output = serializer

    def build_data_pipeline(
        self,
        input: Optional[str] = None,
        deserializer: Optional[Deserializer] = None,
        data_pipeline: Optional[DataPipeline] = None,
    ) -> Optional[DataPipeline]:
        """Build a :class:`.DataPipeline` incorporating available
        :class:`~flash.core.data.io.input_transform.InputTransform` and
        :class:`~flash.core.data.io.output_transform.OutputTransform`
        objects. These will be overridden in the following resolution order (lowest priority first):

        - Lightning ``Datamodule``, either attached to the :class:`.Trainer` or to the :class:`.Task`.
        - :class:`.Task` defaults given to :meth:`.Task.__init__`.
        - :class:`.Task` manual overrides by setting :py:attr:`~data_pipeline`.
        - :class:`.DataPipeline` passed to this method.

        Args:
            input: A string that indicates the format of the data source to use which will override
                the current data source format used.
            deserializer: deserializer to use
            data_pipeline: Optional highest priority source of
                :class:`~flash.core.data.io.input_transform.InputTransform` and
                :class:`~flash.core.data.io.output_transform.OutputTransform`.

        Returns:
            The fully resolved :class:`.DataPipeline`.
        """
        deserializer, old_input, input_transform, output_transform, output = None, None, None, None, None

        # Datamodule
        datamodule = None
        if self.trainer is not None and hasattr(self.trainer, "datamodule"):
            datamodule = self.trainer.datamodule
        elif getattr(self, "datamodule", None) is not None:
            datamodule = self.datamodule

        if getattr(datamodule, "data_pipeline", None) is not None:
            old_input = getattr(datamodule.data_pipeline, "input", None)
            input_transform = getattr(datamodule.data_pipeline, "_input_transform_pipeline", None)
            output_transform = getattr(datamodule.data_pipeline, "_output_transform", None)
            output = getattr(datamodule.data_pipeline, "_output", None)
            deserializer = getattr(datamodule.data_pipeline, "_deserializer", None)

        # Defaults / task attributes
        deserializer, input_transform, output_transform, output = Task._resolve(
            deserializer,
            input_transform,
            output_transform,
            output,
            self._deserializer,
            self._input_transform,
            self._output_transform,
            self._output,
        )

        # Datapipeline
        if data_pipeline is not None:
            deserializer, input_transform, output_transform, output = Task._resolve(
                deserializer,
                input_transform,
                output_transform,
                output,
                getattr(data_pipeline, "_deserializer", None),
                getattr(data_pipeline, "_input_transform_pipeline", None),
                getattr(data_pipeline, "_output_transform", None),
                getattr(data_pipeline, "_output", None),
            )

        input = input or old_input

        if deserializer is None or type(deserializer) is Deserializer:
            deserializer = getattr(input_transform, "deserializer", deserializer)

        data_pipeline = DataPipeline(
            input=input,
            input_transform=input_transform,
            output_transform=output_transform,
            deserializer=deserializer,
            output=output,
        )

        self._data_pipeline_state = self._data_pipeline_state or DataPipelineState()

        self.attach_data_pipeline_state(self._data_pipeline_state)
        self._data_pipeline_state = data_pipeline.initialize(self._data_pipeline_state)
        return data_pipeline

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
        self._deserializer, self._input_transform, self._output_transform, self.output = Task._resolve(
            self._deserializer,
            self._input_transform,
            self._output_transform,
            self._output,
            getattr(data_pipeline, "_deserializer", None),
            getattr(data_pipeline, "_input_transform_pipeline", None),
            getattr(data_pipeline, "_output_transform", None),
            getattr(data_pipeline, "_output", None),
        )

        # self._input_transform.state_dict()
        if getattr(self._input_transform, "_ddp_params_and_buffers_to_ignore", None):
            self._ddp_params_and_buffers_to_ignore = self._input_transform._ddp_params_and_buffers_to_ignore

    @torch.jit.unused
    @property
    def input_transform(self) -> InputTransform:
        return getattr(self.data_pipeline, "_input_transform_pipeline", None)

    @torch.jit.unused
    @property
    def output_transform(self) -> OutputTransform:
        return getattr(self.data_pipeline, "_output_transform", None)

    def on_predict_start(self) -> None:
        self._wrap_predict_step()

    def on_predict_end(self) -> None:
        self._unwrap_predict_step()

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
    def available_backbones(
        cls, head: Optional[str] = None
    ) -> Optional[Union[Dict[str, Optional[List[str]]], List[str]]]:
        if head is None:
            registry: Optional[FlashRegistry] = getattr(cls, "backbones", None)
            if registry is not None and getattr(cls, "heads", None) is None:
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
                backbones = getattr(cls, "backbones", None)
                if backbones is not None:
                    backbones = backbones.available_keys()
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
    def available_optimizers(cls) -> List[str]:
        """Returns a list containing the keys of the available Optimizers."""
        registry: Optional[FlashRegistry] = getattr(cls, "optimizers", None)
        if registry is None:
            return []
        return registry.available_keys()

    @classmethod
    def available_lr_schedulers(cls) -> List[str]:
        """Returns a list containing the keys of the available LR schedulers."""
        registry: Optional[FlashRegistry] = getattr(cls, "lr_schedulers", None)
        if registry is None:
            return []
        return registry.available_keys()

    @classmethod
    def available_finetuning_strategies(cls) -> List[str]:
        """Returns a list containing the keys of the available Finetuning Strategies."""
        registry: Optional[FlashRegistry] = getattr(cls, "finetuning_strategies", None)
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

    def _get_lr_scheduler_class_from_registry(self, lr_scheduler_key: str) -> Dict[str, Any]:
        if lr_scheduler_key.lower() not in self.available_lr_schedulers():
            raise KeyError(
                f"Please provide a valid scheduler name and make sure it is registerd with the Scheduler registry."
                f"\nUse `{self.__class__.__name__}.available_lr_schedulers()` to list the available schedulers."
                f"\n>>> List of available LR Schedulers: {self.available_lr_schedulers()}."
            )
        lr_scheduler_fn: Dict[str, Any] = self.lr_schedulers.get(lr_scheduler_key.lower(), with_metadata=True)
        return deepcopy(lr_scheduler_fn)

    def _instantiate_lr_scheduler(self, optimizer: Optimizer) -> Dict[str, Any]:
        default_scheduler_config = {
            "scheduler": None,
            "name": None,
            "interval": "epoch",
            "frequency": 1,
            "reduce_on_plateau": False,
            "monitor": None,
            "strict": True,
            "opt_idx": None,
        }
        if isinstance(self.lr_scheduler, str):
            lr_scheduler_data: Dict[str, Any] = self._get_lr_scheduler_class_from_registry(self.lr_scheduler)
            lr_scheduler_fn = lr_scheduler_data.pop("fn")
            lr_scheduler_metadata: Dict[str, Any] = lr_scheduler_data.pop("metadata", None)
            lr_scheduler_kwargs: Dict[str, Any] = {}
            lr_scheduler_config = default_scheduler_config
            for key, value in lr_scheduler_config.items():
                lr_scheduler_config[key] = lr_scheduler_metadata.pop(key, None) or value

        elif isinstance(self.lr_scheduler, Callable):
            lr_scheduler_data = {}
            lr_scheduler_fn = self.lr_scheduler
            lr_scheduler_metadata: Dict[str, Any] = None
            lr_scheduler_kwargs: Dict[str, Any] = {}
            lr_scheduler_config = default_scheduler_config

        elif isinstance(self.lr_scheduler, Tuple):
            if len(self.lr_scheduler) not in [2, 3]:
                raise MisconfigurationException(
                    f"The tuple configuration of an scheduler input must be:\n"
                    f"1) Of length 2 with the first index containing a str from {self.available_lr_schedulers()} and"
                    f" the second index containing the required keyword arguments to initialize the LR Scheduler.\n"
                    f"2) Of length 3 with the first index containing a str from {self.available_lr_schedulers()} and"
                    f" the second index containing the required keyword arguments to initialize the LR Scheduler and"
                    f" the third index containing a Lightning scheduler configuration dictionary of the format"
                    f" {default_scheduler_config}. NOTE: Do not set the `scheduler` key in the"
                    f" lr_scheduler_config, it will overridden with an instance of the provided scheduler key."
                )

            if not isinstance(self.lr_scheduler[0], (str, Callable)):
                raise TypeError(
                    f"The first value in lr_scheduler argument tuple should be of type string or type Callable"
                    f" but got {type(self.lr_scheduler[0])}."
                )

            if not isinstance(self.lr_scheduler[1], Dict):
                raise TypeError(
                    f"The second value in lr_scheduler argument tuple should be of type dict but got"
                    f" {type(self.lr_scheduler[1])}."
                )

            if len(self.lr_scheduler) == 3 and not isinstance(self.lr_scheduler[2], Dict):
                raise TypeError(
                    f"The third value in lr_scheduler argument tuple should be of type dict but got"
                    f" {type(self.lr_scheduler[2])}."
                )

            lr_scheduler_data: Dict[str, Any] = self._get_lr_scheduler_class_from_registry(self.lr_scheduler[0])
            lr_scheduler_fn = lr_scheduler_data.pop("fn")
            lr_scheduler_metadata: Dict[str, Any] = lr_scheduler_data.pop("metadata", None)
            lr_scheduler_kwargs: Dict[str, Any] = self.lr_scheduler[1]
            lr_scheduler_config = default_scheduler_config
            for key, value in lr_scheduler_config.items():
                lr_scheduler_config[key] = lr_scheduler_metadata.pop(key, None) or value
            if len(self.lr_scheduler) == 3:
                lr_scheduler_config.update(self.lr_scheduler[2])

        else:
            raise TypeError(
                f"`lr_scheduler` argument should be of type string or callable or tuple(string, dictionary)"
                f" or tuple(string, dictionary, dictionary) but got {type(self.lr_scheduler)}."
            )

        # Providers part
        if lr_scheduler_metadata is not None and "providers" in lr_scheduler_metadata.keys():
            if lr_scheduler_metadata["providers"] == _HUGGINGFACE:
                if lr_scheduler_data["name"] != "constant_schedule":
                    num_training_steps: int = self.get_num_training_steps()
                    num_warmup_steps: int = self._compute_warmup(
                        num_training_steps=num_training_steps,
                        num_warmup_steps=lr_scheduler_kwargs["num_warmup_steps"],
                    )
                    lr_scheduler_kwargs["num_warmup_steps"] = num_warmup_steps
                    if lr_scheduler_data["name"] != "constant_schedule_with_warmup":
                        lr_scheduler_kwargs["num_training_steps"] = num_training_steps

        # User can register a callable that returns a lr_scheduler_config
        # 1) If return value is an instance of _LR_Scheduler -> Add to current config and return the config.
        # 2) If return value is a dictionary, check for the lr_scheduler_config `only keys` and return the config.
        lr_scheduler: Union[_LRScheduler, Dict[str, Any]] = lr_scheduler_fn(optimizer, **lr_scheduler_kwargs)

        if not isinstance(lr_scheduler, (_LRScheduler, Dict)):
            raise MisconfigurationException(
                f"Please make sure that your custom configuration outputs either an LR Scheduler or a scheduler"
                f" configuration with keys belonging to {list(default_scheduler_config.keys())}."
            )

        if isinstance(lr_scheduler, Dict):
            dummy_config = default_scheduler_config
            if not all(config_key in dummy_config.keys() for config_key in lr_scheduler.keys()):
                raise MisconfigurationException(
                    f"Please make sure that your custom configuration outputs either an LR Scheduler or a scheduler"
                    f" configuration with keys belonging to {list(dummy_config.keys())}."
                )
            # If all are present, return the config
            return lr_scheduler

        # If `lr_scheduler` is not a Dict, then add it to the current config and return the config.
        lr_scheduler_config["scheduler"] = lr_scheduler
        return lr_scheduler_config

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if "input_transform.state_dict" in state_dict:
            try:
                input_transform_state_dict = state_dict["input_transform.state_dict"]
                meta = input_transform_state_dict["_meta"]
                cls = getattr(import_module(meta["module"]), meta["class_name"])
                self._input_transform = cls.load_state_dict(
                    {k: v for k, v in input_transform_state_dict.items() if k != "_meta"},
                    strict=strict,
                )
                self._input_transform._state = meta["_state"]
                del state_dict["input_transform.state_dict"]
                del input_transform_state_dict["_meta"]
            except (ModuleNotFoundError, KeyError):
                meta = state_dict["input_transform.state_dict"]["_meta"]
                raise MisconfigurationException(
                    f"The `InputTransform` {meta['module']}.{meta['class_name']}"
                    "has been moved and couldn't be imported."
                )

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def configure_callbacks(self):
        # used only for CI
        if flash._IS_TESTING and torch.cuda.is_available():
            return [BenchmarkConvergenceCI()]

    @requires("serve")
    def run_serve_sanity_check(self, serve_input: ServeInput):
        from fastapi.testclient import TestClient

        from flash.core.serve.flash_components import build_flash_serve_model_component

        print("Running serve sanity check")
        comp = build_flash_serve_model_component(self, serve_input)
        composition = Composition(predict=comp, TESTING=True, DEBUG=True)
        app = composition.serve(host="0.0.0.0", port=8000)

        with TestClient(app) as tc:
            input_str = serve_input.example_input
            body = {"session": "UUID", "payload": {"inputs": {"data": input_str}}}
            resp = tc.post("http://0.0.0.0:8000/predict", json=body)
            print(f"Sanity check response: {resp.json()}")

    @requires("serve")
    def serve(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        sanity_check: bool = True,
        input_cls: Optional[Type[ServeInput]] = None,
        transform: INPUT_TRANSFORM_TYPE = InputTransform,
        transform_kwargs: Optional[Dict] = None,
    ) -> "Composition":
        """Serve the ``Task``. Override this method to provide a default ``input_cls``, ``transform``, and
        ``transform_kwargs``.

        Args:
            host: The IP address to host the ``Task`` on.
            port: The port to host on.
            sanity_check: If ``True``, runs a sanity check before serving.
            input_cls: The ``ServeInput`` type to use.
            transform: The transform to use when serving.
            transform_kwargs: Keyword arguments used to instantiate the transform.
        """
        from flash.core.serve.flash_components import build_flash_serve_model_component

        if input_cls is None:
            raise NotImplementedError("The `input_cls` must be provided to enable serving.")

        serve_input = input_cls(transform=transform, transform_kwargs=transform_kwargs)

        if sanity_check:
            self.run_serve_sanity_check(serve_input)

        comp = build_flash_serve_model_component(self, serve_input)
        composition = Composition(predict=comp, TESTING=flash._IS_TESTING)
        composition.serve(host=host, port=port)
        return composition
