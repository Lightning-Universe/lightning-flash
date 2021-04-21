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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import torch
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import RunningStage
from torch import nn

from flash.core.registry import FlashRegistry
from flash.core.utils import get_callable_dict
from flash.data.data_pipeline import DataPipeline
from flash.data.process import Postprocess, Preprocess, Serializer, SerializerMapping


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
        optimizer: Optimizer to use for training, defaults to `torch.optim.Adam`.
        metrics: Metrics to compute for training and evaluation.
        learning_rate: Learning rate to use for training, defaults to `5e-5`.
        preprocess: :class:`~flash.data.process.Preprocess` to use as the default for this task.
        postprocess: :class:`~flash.data.process.Postprocess` to use as the default for this task.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        loss_fn: Optional[Union[Callable, Mapping, Sequence]] = None,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
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
        self.optimizer_cls = optimizer
        self.metrics = nn.ModuleDict({} if metrics is None else get_callable_dict(metrics))
        self.learning_rate = learning_rate
        # TODO: should we save more? Bug on some regarding yaml if we save metrics
        self.save_hyperparameters("learning_rate", "optimizer")

        self._preprocess: Optional[Preprocess] = preprocess
        self._postprocess: Optional[Postprocess] = postprocess
        self._serializer: Optional[Serializer] = None

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

        data_pipeline = self.build_data_pipeline(data_pipeline)

        x = [x for x in data_pipeline._generate_auto_dataset(x, running_stage)]
        x = data_pipeline.worker_preprocessor(running_stage)(x)
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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer_cls(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)

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
        """Resolves the correct :class:`~flash.data.process.Preprocess`, :class:`~flash.data.process.Postprocess`, and
        :class:`~flash.data.process.Serializer` to use, choosing ``new_*`` if it is not None or a base class
        (:class:`~flash.data.process.Preprocess`, :class:`~flash.data.process.Postprocess`, or
        :class:`~flash.data.process.Serializer`) and ``old_*`` otherwise.

        Args:
            old_preprocess: :class:`~flash.data.process.Preprocess` to be overridden.
            old_postprocess: :class:`~flash.data.process.Postprocess` to be overridden.
            old_serializer: :class:`~flash.data.process.Serializer` to be overridden.
            new_preprocess: :class:`~flash.data.process.Preprocess` to override with.
            new_postprocess: :class:`~flash.data.process.Postprocess` to override with.
            new_serializer: :class:`~flash.data.process.Serializer` to override with.

        Returns:
            The resolved :class:`~flash.data.process.Preprocess`, :class:`~flash.data.process.Postprocess`, and
            :class:`~flash.data.process.Serializer`.
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

    def build_data_pipeline(self, data_pipeline: Optional[DataPipeline] = None) -> Optional[DataPipeline]:
        """Build a :class:`.DataPipeline` incorporating available
        :class:`~flash.data.process.Preprocess` and :class:`~flash.data.process.Postprocess`
        objects. These will be overridden in the following resolution order (lowest priority first):

        - Lightning ``Datamodule``, either attached to the :class:`.Trainer` or to the :class:`.Task`.
        - :class:`.Task` defaults given to ``.Task.__init__``.
        - :class:`.Task` manual overrides by setting :py:attr:`~data_pipeline`.
        - :class:`.DataPipeline` passed to this method.

        Args:
            data_pipeline: Optional highest priority source of
                :class:`~flash.data.process.Preprocess` and :class:`~flash.data.process.Postprocess`.

        Returns:
            The fully resolved :class:`.DataPipeline`.
        """
        preprocess, postprocess, serializer = None, None, None

        # Datamodule
        if self.datamodule is not None and getattr(self.datamodule, 'data_pipeline', None) is not None:
            preprocess = getattr(self.datamodule.data_pipeline, '_preprocess_pipeline', None)
            postprocess = getattr(self.datamodule.data_pipeline, '_postprocess_pipeline', None)
            serializer = getattr(self.datamodule.data_pipeline, '_serializer', None)

        elif self.trainer is not None and hasattr(
            self.trainer, 'datamodule'
        ) and getattr(self.trainer.datamodule, 'data_pipeline', None) is not None:
            preprocess = getattr(self.trainer.datamodule.data_pipeline, '_preprocess_pipeline', None)
            postprocess = getattr(self.trainer.datamodule.data_pipeline, '_postprocess_pipeline', None)
            serializer = getattr(self.trainer.datamodule.data_pipeline, '_serializer', None)

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

        data_pipeline = DataPipeline(preprocess, postprocess, serializer)
        data_pipeline.initialize()
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

            if 'data_pipeline_state' not in checkpoint:
                checkpoint['data_pipeline_state'] = checkpoint['data_pipeline'].state
        super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        if 'data_pipeline' in checkpoint:
            data_pipeline = checkpoint['data_pipeline']

            if 'data_pipeline_state' in checkpoint:
                data_pipeline.state = checkpoint['data_pipeline_state']
            self.data_pipeline = data_pipeline

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
