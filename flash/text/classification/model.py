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
from typing import Any, Dict, List, Optional, Type, Union

from pytorch_lightning import Callback

from flash.core.classification import ClassificationAdapterTask
from flash.core.data.io.input import ServeInput
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.io.output import Output
from flash.core.registry import FlashRegistry
from flash.core.serve import Composition
from flash.core.utilities.imports import requires
from flash.core.utilities.types import (
    INPUT_TRANSFORM_TYPE,
    LOSS_FN_TYPE,
    LR_SCHEDULER_TYPE,
    METRICS_TYPE,
    OPTIMIZER_TYPE,
)
from flash.text.classification.backbones import TEXT_CLASSIFIER_BACKBONES
from flash.text.input import TextDeserializer
from flash.text.ort_callback import ORTCallback


class TextClassifier(ClassificationAdapterTask):
    """The ``TextClassifier`` is a :class:`~flash.Task` for classifying text. For more details, see
    :ref:`text_classification`. The ``TextClassifier`` also supports multi-label classification with
    ``multi_label=True``. For more details, see :ref:`text_classification_multi_label`.

    Args:
        num_classes: Number of classes to classify.
        backbone: A model to use to compute text features can be any BERT model from HuggingFace/transformersimage.
        max_length: The maximum length to pad / truncate sequences to.
        optimizer: Optimizer to use for training.
        lr_scheduler: The LR scheduler to use during training.
        metrics: Metrics to compute for training and evaluation. Can either be an metric from the `torchmetrics`
            package, a custom metric inherenting from `torchmetrics.Metric`, a callable function or a list/dict
            containing a combination of the aforementioned. In all cases, each metric needs to have the signature
            `metric(preds,target)` and return a single scalar tensor. Defaults to :class:`torchmetrics.Accuracy`.
        learning_rate: Learning rate to use for training, defaults to `1e-3`
        multi_label: Whether the targets are multi-label or not.
        enable_ort: Enable Torch ONNX Runtime Optimization: https://onnxruntime.ai/docs/#onnx-runtime-for-training
    """

    required_extras: str = "text"

    backbones: FlashRegistry = TEXT_CLASSIFIER_BACKBONES

    def __init__(
        self,
        num_classes: Optional[int] = None,
        labels: Optional[List[str]] = None,
        backbone: str = "prajjwal1/bert-medium",
        max_length: int = 128,
        loss_fn: LOSS_FN_TYPE = None,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        metrics: METRICS_TYPE = None,
        learning_rate: Optional[float] = None,
        multi_label: bool = False,
        enable_ort: bool = False,
        **kwargs,
    ):
        self.save_hyperparameters()

        if labels is not None and num_classes is None:
            num_classes = len(labels)

        metadata = self.backbones.get(backbone, with_metadata=True)
        adapter = metadata["metadata"]["adapter"].from_task(
            self,
            backbone=metadata["fn"],
            num_classes=num_classes,
            max_length=max_length,
            **kwargs,
        )

        super().__init__(
            adapter,
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            multi_label=multi_label,
            num_classes=num_classes,
            labels=labels,
        )

        self.enable_ort = enable_ort
        self.max_length = max_length

    def _ci_benchmark_fn(self, history: List[Dict[str, Any]]):
        """This function is used only for debugging usage with CI."""
        if self.hparams.multi_label:
            assert history[-1]["val_f1score"] > 0.40, history[-1]["val_f1score"]
        else:
            assert history[-1]["val_accuracy"] > 0.70, history[-1]["val_accuracy"]

    def configure_callbacks(self) -> List[Callback]:
        callbacks = super().configure_callbacks() or []
        if self.enable_ort:
            callbacks.append(ORTCallback())
        return callbacks

    @requires("serve")
    def serve(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        sanity_check: bool = True,
        input_cls: Optional[Type[ServeInput]] = TextDeserializer,
        transform: INPUT_TRANSFORM_TYPE = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        output: Optional[Union[str, Output]] = None,
    ) -> Composition:
        return super().serve(host, port, sanity_check, input_cls, transform, transform_kwargs, output)
