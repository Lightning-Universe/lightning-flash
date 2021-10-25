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
from copy import deepcopy
from importlib import import_module
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import torch
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

import flash
from flash.core.data_v2.data_module import DataModule
from flash.core.data_v2.data_pipeline import DataPipeline, DataPipelineState
from flash.core.data_v2.io.base_input import BaseInput
from flash.core.data_v2.io.input import InputFormat, InputsStateContainer
from flash.core.data_v2.io.output import Output
from flash.core.data_v2.transforms.output_transform import OutputTransform
from flash.core.model import (
    BenchmarkConvergenceCI,
    CheckDependenciesMeta,
    DatasetProcessor,
    ModuleWrapperBase,
    OutputKeys,
    predict_context,
)
from flash.core.registry import FlashRegistry
from flash.core.schedulers import _SCHEDULERS_REGISTRY
from flash.core.serve import Composition
from flash.core.utilities.apply_func import get_callable_dict
from flash.core.utilities.imports import requires
from flash.core.utilities.stages import RunningStage


class TaskV2(DatasetProcessor, ModuleWrapperBase, LightningModule, metaclass=CheckDependenciesMeta):
    """A general Task.

    Args:
        model: Model to use for the task.
        loss_fn: Loss function for training.
        optimizer: Optimizer to use for training, defaults to :class:`torch.optim.Adam`.
        optimizer_kwargs: A dict containing additional arguments for initializing the optimizer. Note that
            this includes all other arguments than the learning rate which is a seperate argument.
        scheduler: Learning rate scheduler to use for the training. Should be an instance of
            :class:`torch.optim.lr_scheduler._LRScheduler`. Defaults to using no scheduler.
        scheduler_kwargs: A dict containing additional arguments for initializing the learning rate scheduler.
        metrics: Metrics to compute for training and evaluation. Can either be an metric from the `torchmetrics`
            package, a custom metric inheriting from `torchmetrics.Metric`, a callable function or a list/dict
            containing a combination of the aforementioned. In all cases, each metric needs to have the signature
            `metric(preds,target)` and return a single scalar tensor.
        learning_rate: Learning rate to use for training, defaults to ``5e-5``.
        deserializer: Either a single :class:`~flash.core.data.process.Deserializer` or a mapping of these to
            deserialize the input
        preprocess: :class:`~flash.core.data.process.Preprocess` to use as the default for this task.
        postprocess: :class:`~flash.core.data.process.Postprocess` to use as the default for this task.
        serializer: Either a single :class:`~flash.core.data.process.Serializer` or a mapping of these to
            serialize the output e.g. convert the model output into the desired output format when predicting.
    """

    _datamodule_cls: Optional[DataModule] = None
    _inputs_state: Optional[InputsStateContainer] = None
    _inputs_registry: Optional[FlashRegistry] = None
    _output_transform: Optional[OutputTransform] = None
    _output: Optional[Output] = None
    _output_registry = FlashRegistry("output")

    schedulers: FlashRegistry = _SCHEDULERS_REGISTRY
    required_extras: Optional[Union[str, List[str]]] = None

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
        output_transform: Optional[OutputTransform] = OutputTransform(),
        output: Optional[Union[Output, Mapping[str, Output]]] = None,
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

        self.output_transform = output_transform
        self.output = self._output_registry.get(output)() if isinstance(output, str) else output

        self.predict_step = self._wrap_predict_step(self.predict_step)

    @property
    def output(self) -> Optional[Output]:
        return self._output

    @output.setter
    def output(self, output: Optional[Output]) -> None:
        self._output = output

    @property
    def output_transform(self) -> Optional[Output]:
        return self._output_transform

    @output_transform.setter
    def output_transform(self, output_transform: Optional[OutputTransform]) -> None:
        self._output_transform = output_transform

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
            if isinstance(metric, torchmetrics.Metric):
                metric(y_hat, y)
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
        output = self.step(batch, batch_idx, self.val_metrics)
        self.log_dict(
            {f"test_{k}": v for k, v in output[OutputKeys.LOGS].items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(self) -> Union[Optimizer, Tuple[List[Optimizer], List[_LRScheduler]]]:
        """Implement how optimizer and optionally learning rate schedulers should be configured."""
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

    @torch.jit.unused
    @property
    def data_pipeline(self) -> DataPipeline:
        data_pipeline = DataPipeline(
            inputs_registry=self._inputs_registry,
            inputs_state=self._inputs_state,
            output_transform=self._output_transform,
            output=self._output,
        )
        self._data_pipeline_state = self._data_pipeline_state or DataPipelineState()
        self.attach_data_pipeline_state(self._data_pipeline_state)
        self._data_pipeline_state = data_pipeline.initialize(self._data_pipeline_state)
        return data_pipeline

    @torch.jit.unused
    @data_pipeline.setter
    def data_pipeline(self, data_pipeline: Optional[DataPipeline]) -> None:
        self._resolve_inputs_registry(data_pipeline._inputs_registry)
        self._resolve_inputs_state(data_pipeline._inputs_state)
        self._resolve_inputs_state(data_pipeline._output_transform)

    def _resolve_inputs_registry(self, inputs_registry: Optional[FlashRegistry]) -> None:
        if not self._inputs_registry:
            self._inputs_registry = inputs_registry or self._datamodule_cls.inputs_registry

    def _resolve_inputs_state(self, inputs_state: Optional[InputsStateContainer]) -> None:
        if not self._inputs_state:
            self._inputs_state = inputs_state

    def _resolve_output_transform(self, output_transform: Optional[OutputTransform]) -> None:
        if not self._output_transform:
            self._output_transform = output_transform

    def _resolve_transform(self, input_transform_stage: RunningStage):
        state = None
        if input_transform_stage == RunningStage.TRAINING:
            state = self._inputs_state.train_input_state
        elif input_transform_stage == RunningStage.VALIDATING:
            state = self._inputs_state.val_input_state
        elif input_transform_stage == RunningStage.TESTING:
            state = self._inputs_state.test_input_state
        elif input_transform_stage == RunningStage.PREDICTING:
            state = self._inputs_state.predict_input_state
        return state.input_transform if state else state

    def _wrap_predict_step(self, predict_step: Callable) -> Callable:
        @functools.wraps(predict_step)
        def wrapped_func(*args: Any, **kwargs: Any) -> Optional[Any]:
            predictions = predict_step(*args, **kwargs)
            if self.output_transform:
                predictions = self.output_transform(predictions)
            if self.output:
                predictions = [self.output(sample) for sample in predictions]

            if isinstance(predictions, torch.Tensor) and isinstance(predictions[0], torch.Tensor):
                return torch.stack(predictions)
            return predictions

        return wrapped_func

    @predict_context
    def predict(
        self,
        x: Any,
        input: Optional[str] = InputFormat.DEFAULT,
        input_transform_stage: RunningStage = RunningStage.TRAINING,
        output: Optional[str] = None,
        running_stage: Optional[RunningStage] = RunningStage.PREDICTING,
    ) -> Any:
        """Predict function for raw data or processed data.

        Args:
            x: Input to predict. Can be raw data or processed data. If str, assumed to be a folder of data.
            data_source: A string that indicates the format of the data source to use which will override
                the current data source format used
            deserializer: A single :class:`~flash.core.data.process.Deserializer` to deserialize the input
            data_pipeline: Use this to override the current data pipeline

        Returns:
            The post-processed model predictions
        """
        assert running_stage in (RunningStage.PREDICTING, RunningStage.SERVING)

        transform = self._resolve_transform(input_transform_stage)
        self.output = self._output_registry.get(output)() if isinstance(output, str) else output
        input_cls: BaseInput = self.data_pipeline._inputs_registry.get(input)
        input = input_cls.from_data(x, running_stage=running_stage, transform=transform)
        x = input.dataloader_collate_fn([x for x in input])
        x = self.transfer_batch_to_device(x, self.device, 0)
        x = input.on_after_batch_transfer_fn(x)
        x = x[0] if isinstance(x, list) else x
        return self.predict_step(x, 0)

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

    @requires("serve")
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

    @requires("serve")
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
