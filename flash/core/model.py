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
import inspect
import re
from abc import ABCMeta
from copy import deepcopy
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning.utilities.enums import LightningEnum
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Sampler

import flash
from flash.core.data.io.input import InputBase, ServeInput
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.io.output import Output
from flash.core.data.io.output_transform import OutputTransform
from flash.core.data.output import BASE_OUTPUTS
from flash.core.finetuning import _DEFAULTS_FINETUNE_STRATEGIES, _FINETUNING_STRATEGIES_REGISTRY
from flash.core.hooks import FineTuningHooks
from flash.core.optimizers.optimizers import _OPTIMIZERS_REGISTRY
from flash.core.optimizers.schedulers import _SCHEDULERS_REGISTRY
from flash.core.registry import FlashRegistry
from flash.core.serve.composition import Composition
from flash.core.utilities.apply_func import get_callable_dict
from flash.core.utilities.imports import _PL_GREATER_EQUAL_1_5_0, requires
from flash.core.utilities.providers import _HUGGINGFACE
from flash.core.utilities.types import (
    INPUT_TRANSFORM_TYPE,
    LOSS_FN_TYPE,
    LR_SCHEDULER_TYPE,
    METRICS_TYPE,
    MODEL_TYPE,
    OPTIMIZER_TYPE,
    OUTPUT_TRANSFORM_TYPE,
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

    def __setattr__(self, key, value):
        if isinstance(value, (LightningModule, ModuleWrapperBase)):
            self._children.append(key)
        patched_attributes = ["_current_fx_name", "_current_hook_fx_name", "_results", "_data_pipeline_state"]
        if isinstance(value, Trainer) or key in patched_attributes:
            if hasattr(self, "_children"):
                for child in self._children:
                    setattr(getattr(self, child), key, value)
        super().__setattr__(key, value)


class DatasetProcessor:
    """The ``DatasetProcessor`` mixin provides hooks for classes which need custom logic for producing the data
    loaders for each running stage given the corresponding dataset."""

    def __init__(self):
        super().__init__()

        self._collate_fn = None
        self._input_transform = None

    @torch.jit.unused
    @property
    def collate_fn(self) -> Optional[Callable]:
        return self._collate_fn

    @collate_fn.setter
    def collate_fn(self, collate_fn: Callable) -> None:
        self._collate_fn = collate_fn

    @torch.jit.unused
    @property
    def input_transform(self) -> Optional[INPUT_TRANSFORM_TYPE]:
        return self._input_transform

    @input_transform.setter
    def input_transform(self, input_transform: INPUT_TRANSFORM_TYPE) -> None:
        self._input_transform = input_transform

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
        persistent_workers: bool = False,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
            collate_fn=self.collate_fn if self.collate_fn is not None else collate_fn,
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
        persistent_workers: bool = False,
    ) -> DataLoader:
        return self._process_dataset(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn if self.collate_fn is not None else collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
            persistent_workers=persistent_workers,
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
        persistent_workers: bool = False,
    ) -> DataLoader:
        return self._process_dataset(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn if self.collate_fn is not None else collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
            persistent_workers=persistent_workers,
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
        persistent_workers: bool = False,
    ) -> DataLoader:
        return self._process_dataset(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn if self.collate_fn is not None else collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
            persistent_workers=persistent_workers,
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
        persistent_workers: bool = False,
    ) -> DataLoader:
        return self._process_dataset(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn if self.collate_fn is not None else collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
            persistent_workers=persistent_workers,
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
        learning_rate: Learning rate to use for training. If ``None`` (the default) then the default LR for your chosen
            optimizer will be used.
        optimizer: Optimizer to use for training.
        lr_scheduler: The LR scheduler to use during training.
        metrics: Metrics to compute for training and evaluation. Can either be an metric from the `torchmetrics`
            package, a custom metric inheriting from `torchmetrics.Metric`, a callable function or a list/dict
            containing a combination of the aforementioned. In all cases, each metric needs to have the signature
            `metric(preds,target)` and return a single scalar tensor.
        output_transform: :class:`~flash.core.data.io.output_transform.OutputTransform` to use as the default for this
            task.
    """

    optimizers: FlashRegistry = _OPTIMIZERS_REGISTRY
    lr_schedulers: FlashRegistry = _SCHEDULERS_REGISTRY
    finetuning_strategies: FlashRegistry = _FINETUNING_STRATEGIES_REGISTRY
    outputs: FlashRegistry = BASE_OUTPUTS

    required_extras: Optional[Union[str, List[str]]] = None

    def __init__(
        self,
        model: MODEL_TYPE = None,
        loss_fn: LOSS_FN_TYPE = None,
        learning_rate: Optional[float] = None,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        metrics: METRICS_TYPE = None,
        output_transform: OUTPUT_TRANSFORM_TYPE = None,
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

        self._output_transform: Optional[OutputTransform] = output_transform

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
        optimizers_kwargs: Dict[str, Any] = {}
        if isinstance(self.optimizer, str):
            optimizer_fn = self._get_optimizer_class_from_registry(self.optimizer.lower())
        elif isinstance(self.optimizer, Callable):
            optimizer_fn = self.optimizer
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
        else:
            raise TypeError(
                f"""Optimizer should be of type string or callable or tuple(string, dictionary)
                but got {type(self.optimizer)}."""
            )

        if self.learning_rate is not None:
            optimizers_kwargs["lr"] = self.learning_rate

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
                    f"Please provide a valid strategy from {_DEFAULTS_FINETUNE_STRATEGIES[:2]}."
                    " For more details and advanced finetuning options see our docs:"
                    " https://lightning-flash.readthedocs.io/en/stable/general/finetuning.html"
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
    def run_serve_sanity_check(self, serve_input: ServeInput, output: Output):
        from fastapi.testclient import TestClient

        from flash.core.serve.flash_components import build_flash_serve_model_component

        print("Running serve sanity check")
        comp = build_flash_serve_model_component(self, serve_input, output)
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
        output: Optional[Union[str, Output]] = None,
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

        output = output or Output()
        if isinstance(output, str):
            output = self.outputs.get(output).from_task(self)

        if sanity_check:
            self.run_serve_sanity_check(serve_input, output)

        comp = build_flash_serve_model_component(self, serve_input, output)
        composition = Composition(predict=comp, TESTING=flash._IS_TESTING)
        composition.serve(host=host, port=port)
        return composition
