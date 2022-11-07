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
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning.utilities.enums import LightningEnum
from torch import nn, Tensor
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Sampler

import flash
from flash.core.data.io.input import InputBase, ServeInput
from flash.core.data.io.input_transform import (
    create_or_configure_input_transform,
    create_worker_input_transform_processor,
    InputTransform,
)
from flash.core.data.io.output import Output
from flash.core.data.io.output_transform import OutputTransform
from flash.core.data.output import BASE_OUTPUTS
from flash.core.data.utilities.collate import default_collate
from flash.core.finetuning import _FINETUNING_STRATEGIES_REGISTRY
from flash.core.hooks import FineTuningHooks
from flash.core.optimizers.optimizers import _OPTIMIZERS_REGISTRY
from flash.core.optimizers.schedulers import _SCHEDULERS_REGISTRY
from flash.core.registry import FlashRegistry
from flash.core.serve.composition import Composition
from flash.core.utilities.apply_func import get_callable_dict
from flash.core.utilities.imports import _CORE_TESTING, _PL_GREATER_EQUAL_1_5_0, requires
from flash.core.utilities.providers import _HUGGINGFACE
from flash.core.utilities.stages import RunningStage
from flash.core.utilities.types import (
    INPUT_TRANSFORM_TYPE,
    LOSS_FN_TYPE,
    LR_SCHEDULER_TYPE,
    METRICS_TYPE,
    MODEL_TYPE,
    OPTIMIZER_TYPE,
    OUTPUT_TRANSFORM_TYPE,
)

# Skip doctests if requirements aren't available
if not _CORE_TESTING:
    __doctest_skip__ = ["Task", "Task.*"]


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

        self._collate_fn = default_collate
        self._input_transform = None

    @torch.jit.unused
    @property
    def collate_fn(self) -> Callable:
        return self._collate_fn

    @collate_fn.setter
    def collate_fn(self, collate_fn: Callable) -> None:
        self._collate_fn = collate_fn

    @torch.jit.unused
    @property
    def input_transform(self) -> Optional[InputTransform]:
        if self._input_transform is not None:
            return create_or_configure_input_transform(self._input_transform)
        return None

    @input_transform.setter
    def input_transform(self, input_transform: InputTransform) -> None:
        self._input_transform = input_transform

    def process_train_dataset(
        self,
        dataset: InputBase,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = True,
        drop_last: bool = True,
        sampler: Optional[Sampler] = None,
        persistent_workers: bool = False,
        input_transform: Optional[InputTransform] = None,
        trainer: Optional["flash.Trainer"] = None,
    ) -> DataLoader:
        input_transform = input_transform or self.input_transform

        collate_fn = self.collate_fn
        if input_transform is not None:
            # Inject the `self.collate_fn`
            input_transform.inject_collate_fn(self.collate_fn)

            collate_fn = create_worker_input_transform_processor(RunningStage.TRAINING, input_transform)

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

    def process_val_dataset(
        self,
        dataset: InputBase,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = False,
        drop_last: bool = False,
        sampler: Optional[Sampler] = None,
        persistent_workers: bool = False,
        input_transform: Optional[InputTransform] = None,
        trainer: Optional["flash.Trainer"] = None,
    ) -> DataLoader:
        input_transform = input_transform or self.input_transform

        collate_fn = self.collate_fn
        if input_transform is not None:
            # Inject the `self.collate_fn`
            input_transform.inject_collate_fn(self.collate_fn)

            collate_fn = create_worker_input_transform_processor(RunningStage.VALIDATING, input_transform)

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

    def process_test_dataset(
        self,
        dataset: InputBase,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = False,
        drop_last: bool = False,
        sampler: Optional[Sampler] = None,
        persistent_workers: bool = False,
        input_transform: Optional[InputTransform] = None,
        trainer: Optional["flash.Trainer"] = None,
    ) -> DataLoader:
        input_transform = input_transform or self.input_transform

        collate_fn = self.collate_fn
        if input_transform is not None:
            # Inject the `self.collate_fn`
            input_transform.inject_collate_fn(self.collate_fn)

            collate_fn = create_worker_input_transform_processor(RunningStage.TESTING, input_transform)

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

    def process_predict_dataset(
        self,
        dataset: InputBase,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = False,
        drop_last: bool = False,
        sampler: Optional[Sampler] = None,
        persistent_workers: bool = False,
        input_transform: Optional[InputTransform] = None,
        trainer: Optional["flash.Trainer"] = None,
    ) -> DataLoader:
        input_transform = input_transform or self.input_transform

        collate_fn = self.collate_fn
        if input_transform is not None:
            # Inject the `self.collate_fn`
            input_transform.inject_collate_fn(self.collate_fn)

            collate_fn = create_worker_input_transform_processor(RunningStage.PREDICTING, input_transform)

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
            if isinstance(result.required_extras, str):
                result.required_extras = [result.required_extras]
            result.__init__ = requires(*result.required_extras)(result.__init__)

            patterns = ["load_from_checkpoint", "available_*"]  # must match classmethods only
            regex = "(" + ")|(".join(patterns) + ")"
            for attribute_name, attribute_value in filter(lambda x: re.match(regex, x[0]), inspect.getmembers(result)):
                # TODO: Find a better way to do this
                if attribute_name in ["available_layers"]:
                    continue
                setattr(
                    result, attribute_name, classmethod(requires(*result.required_extras)(attribute_value.__func__))
                )
        return result


class OutputKeys(LightningEnum):
    """The ``OutputKeys`` enum contains the keys that are used internally by the ``Task`` when handling outputs."""

    OUTPUT = "y_hat"
    TARGET = "y"
    LOGS = "logs"
    LOSS = "loss"
    BATCH_SIZE = "batch_size"

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

    optimizers_registry: FlashRegistry = _OPTIMIZERS_REGISTRY
    lr_schedulers_registry: FlashRegistry = _SCHEDULERS_REGISTRY
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
        self.save_hyperparameters("learning_rate", "optimizer", ignore=["model", "backbone", "head", "adapter"])

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
        output[OutputKeys.BATCH_SIZE] = y.shape[0] if isinstance(y, Tensor) else None
        return output

    def compute_loss(self, losses: Dict[str, Tensor]) -> Tensor:
        return list(losses.values())[0]

    def compute_logs(self, logs: Dict[str, Any], losses: Dict[str, Tensor]):
        logs.update(losses)
        return logs

    @staticmethod
    def apply_filtering(y: Tensor, y_hat: Tensor) -> Tuple[Tensor, Tensor]:
        """This function is used to filter some labels or predictions which aren't conform."""
        return y, y_hat

    @staticmethod
    def to_loss_format(x: Tensor) -> Tensor:
        return x

    @staticmethod
    def to_metrics_format(x: Tensor) -> Tensor:
        return x

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        output = self.step(batch, batch_idx, self.train_metrics)
        log_kwargs = {"batch_size": output.get(OutputKeys.BATCH_SIZE, None)} if _PL_GREATER_EQUAL_1_5_0 else {}
        self.log_dict(
            {f"train_{k}": v for k, v in output[OutputKeys.LOGS].items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            **log_kwargs,
        )
        return output[OutputKeys.LOSS]

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        output = self.step(batch, batch_idx, self.val_metrics)
        log_kwargs = {"batch_size": output.get(OutputKeys.BATCH_SIZE, None)} if _PL_GREATER_EQUAL_1_5_0 else {}
        self.log_dict(
            {f"val_{k}": v for k, v in output[OutputKeys.LOGS].items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            **log_kwargs,
        )

    def test_step(self, batch: Any, batch_idx: int) -> None:
        output = self.step(batch, batch_idx, self.test_metrics)
        log_kwargs = {"batch_size": output.get(OutputKeys.BATCH_SIZE, None)} if _PL_GREATER_EQUAL_1_5_0 else {}
        self.log_dict(
            {f"test_{k}": v for k, v in output[OutputKeys.LOGS].items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            **log_kwargs,
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

    def modules_to_freeze(self) -> Optional[nn.Module]:
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
        optimizer_fn = self.optimizers_registry.get(optimizer_key.lower())
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
                raise TypeError(
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
                raise ValueError(
                    f"The `strategy` should be one of: {', '.join(self.available_finetuning_strategies())}."
                    " For more details and advanced finetuning options see our docs:"
                    " https://lightning-flash.readthedocs.io/en/stable/general/finetuning.html"
                )
            finetuning_strategy_fn: Callable = self.finetuning_strategies.get(key=strategy)
            finetuning_strategy_metadata = {"strategy_metadata": None, "train_bn": train_bn}
        elif isinstance(strategy, Tuple):
            if not isinstance(strategy[0], str) or strategy[0] not in self.available_finetuning_strategies():
                raise TypeError(
                    f"The first input of `strategy` in a tuple configuration should be one of:"
                    f" {', '.join(self.available_finetuning_strategies())}."
                )
            finetuning_strategy_fn: Callable = self.finetuning_strategies.get(key=strategy[0])
            finetuning_strategy_metadata = {"strategy_metadata": strategy[1], "train_bn": train_bn}
        else:
            raise TypeError(
                "The `strategy` should be a ``pytorch_lightning.callbacks.BaseFinetuning`` callback or one of: "
                f"{', '.join(self.available_finetuning_strategies())}."
            )

        return [finetuning_strategy_fn(**finetuning_strategy_metadata)]

    def as_embedder(self, layer: str):
        """Convert this task to an embedder. Note that the parameters are not copied so that any optimization of
        the embedder will also apply to the converted ``Task``.

        Args:
            layer: The layer to embed to. This should be one of the :meth:`~flash.core.model.Task.available_layers`.
        """
        from flash.core.utilities.embedder import Embedder  # Avoid circular import

        return Embedder(self, layer)

    def available_layers(self):
        """Get the list of available layers for use with the :meth:`~flash.core.model.Task.as_embedder` method."""
        available_layers = []
        for name, _ in self.named_modules():
            if name not in ["train_metrics", "val_metrics", "test_metrics"]:
                available_layers.append(name)
        return ["output"] + available_layers

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
        registry: Optional[FlashRegistry] = getattr(cls, "optimizers_registry", None)
        if registry is None:
            return []
        return registry.available_keys()

    @classmethod
    def available_lr_schedulers(cls) -> List[str]:
        """Returns a list containing the keys of the available LR schedulers."""
        registry: Optional[FlashRegistry] = getattr(cls, "lr_schedulers_registry", None)
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

    @classmethod
    def available_outputs(cls) -> List[str]:
        """Returns the list of available outputs (that can be used during prediction or serving) for this ``Task``.

        Examples
        ________

        ..testsetup::

            >>> from flash import Task

        .. doctest::

            >>> print(Task.available_outputs())
            ['preds', 'raw']
        """
        return cls.outputs.available_keys()

    def get_num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if not getattr(self, "trainer", None):
            raise RuntimeError("The LightningModule isn't attached to the trainer yet.")

        if hasattr(self.trainer, "estimated_stepping_batches"):
            return self.trainer.estimated_stepping_batches

        from flash.core.trainer import Trainer

        return Trainer.estimated_stepping_batches.fget(self.trainer)

    @staticmethod
    def _compute_warmup(num_training_steps: int, num_warmup_steps: Union[int, float]) -> int:
        if not isinstance(num_warmup_steps, float) or (num_warmup_steps > 1 or num_warmup_steps < 0):
            raise TypeError("`num_warmup_steps` should be provided as float between 0 and 1 in `scheduler_kwargs`")
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
        lr_scheduler_fn: Dict[str, Any] = self.lr_schedulers_registry.get(lr_scheduler_key.lower(), with_metadata=True)
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
                raise TypeError(
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

        if isinstance(lr_scheduler, Dict):
            dummy_config = default_scheduler_config
            if not all(config_key in dummy_config.keys() for config_key in lr_scheduler.keys()):
                raise ValueError(
                    f"Please make sure that your custom configuration outputs either an LR Scheduler or a scheduler"
                    f" configuration with keys belonging to {list(dummy_config.keys())}."
                )
            # If all are present, return the config
            return lr_scheduler

        # If `lr_scheduler` is not a Dict, then add it to the current config and return the config.
        lr_scheduler_config["scheduler"] = lr_scheduler
        return lr_scheduler_config

    def configure_callbacks(self):
        # used only for CI
        if flash._IS_TESTING and torch.cuda.is_available():
            return [BenchmarkConvergenceCI()]

    @requires("serve")
    def run_serve_sanity_check(
        self, serve_input: ServeInput, transform: INPUT_TRANSFORM_TYPE, transform_kwargs: Optional[Dict], output: Output
    ):
        from fastapi.testclient import TestClient

        from flash.core.serve.flash_components import build_flash_serve_model_component

        print("Running serve sanity check")
        comp = build_flash_serve_model_component(self, serve_input, output, transform, transform_kwargs)
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

        serve_input = input_cls()

        output = output or Output()
        if isinstance(output, str):
            output = self.outputs.get(output).from_task(self)

        if sanity_check:
            self.run_serve_sanity_check(serve_input, transform, transform_kwargs, output)

        comp = build_flash_serve_model_component(self, serve_input, output, transform, transform_kwargs)
        composition = Composition(predict=comp, TESTING=flash._IS_TESTING)
        composition.serve(host=host, port=port)
        return composition
