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
from types import FunctionType
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from pytorch_lightning import Callback
from torch import nn

from flash.core.classification import ClassificationTask, Labels
from flash.core.data.data_source import DefaultDataKeys
from flash.core.registry import FlashRegistry
from flash.core.utilities.types import LOSS_FN_TYPE, LR_SCHEDULER_TYPE, METRICS_TYPE, OPTIMIZER_TYPE, SERIALIZER_TYPE
from flash.text.classification.backbones import TEXT_CLASSIFIER_BACKBONES
from flash.text.ort_callback import ORTCallback


class Model(torch.nn.Module):
    def __init__(self, backbone: nn.Module, head: Optional[nn.Module]):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        if self.head is None:
            return x
        return self.head(x)


class TextClassifier(ClassificationTask):
    """The ``TextClassifier`` is a :class:`~flash.Task` for classifying text. For more details, see
    :ref:`text_classification`. The ``TextClassifier`` also supports multi-label classification with
    ``multi_label=True``. For more details, see :ref:`text_classification_multi_label`.

    Args:
        num_classes: Number of classes to classify.
        backbone: A model to use to compute text features can be any BERT model from HuggingFace/transformersimage .
        optimizer: Optimizer to use for training.
        lr_scheduler: The LR scheduler to use during training.
        metrics: Metrics to compute for training and evaluation. Can either be an metric from the `torchmetrics`
            package, a custom metric inherenting from `torchmetrics.Metric`, a callable function or a list/dict
            containing a combination of the aforementioned. In all cases, each metric needs to have the signature
            `metric(preds,target)` and return a single scalar tensor. Defaults to :class:`torchmetrics.Accuracy`.
        learning_rate: Learning rate to use for training, defaults to `1e-3`
        multi_label: Whether the targets are multi-label or not.
        serializer: The :class:`~flash.core.data.process.Serializer` to use when serializing prediction outputs.
        enable_ort: Enable Torch ONNX Runtime Optimization: https://onnxruntime.ai/docs/#onnx-runtime-for-training
    """

    required_extras: str = "text"

    backbones: FlashRegistry = TEXT_CLASSIFIER_BACKBONES

    def __init__(
        self,
        num_classes: int,
        backbone: Union[str, Tuple[nn.Module, int]] = "prajjwal1/bert-tiny",
        backbone_kwargs: Optional[Dict] = None,
        pretrained: bool = True,
        head: Optional[Union[FunctionType, nn.Module]] = None,
        loss_fn: Optional[LOSS_FN_TYPE] = None,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: Optional[LR_SCHEDULER_TYPE] = None,
        metrics: Optional[METRICS_TYPE] = None,
        learning_rate: float = 1e-2,
        multi_label: bool = False,
        serializer: Optional[SERIALIZER_TYPE] = None,
        enable_ort: bool = False,
    ):
        self.save_hyperparameters()

        self.enable_ort = enable_ort
        self.pretrained = pretrained

        if not backbone_kwargs:
            backbone_kwargs = {}

        if isinstance(backbone, tuple):
            backbone, num_features = backbone
        else:
            backbone, num_features = self.backbones.get(backbone)(pretrained=pretrained, **backbone_kwargs)

        head = head(num_features, num_classes) if isinstance(head, FunctionType) else head
        head = head or nn.Sequential(
            nn.Linear(num_features, num_classes),
        )

        super().__init__(
            num_classes=num_classes,
            model=Model(backbone=backbone, head=head),
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            learning_rate=learning_rate,
            multi_label=multi_label,
            serializer=serializer or Labels(multi_label=multi_label),
        )

    @property
    def backbone(self):
        return self.model.backbone

    def forward(self, batch: Dict[str, torch.Tensor]):
        return self.model(batch)

    def step(self, batch, batch_idx, metrics) -> dict:
        target = batch.pop(DefaultDataKeys.TARGET)
        batch = (batch, target)
        return super().step(batch, batch_idx, metrics)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self(batch)

    def configure_callbacks(self) -> List[Callback]:
        callbacks = super().configure_callbacks() or []
        if self.enable_ort:
            callbacks.append(ORTCallback())
        return callbacks

    def _ci_benchmark_fn(self, history: List[Dict[str, Any]]):
        """This function is used only for debugging usage with CI."""
        if self.hparams.multi_label:
            assert history[-1]["val_f1"] > 0.40, history[-1]["val_f1"]
        else:
            assert history[-1]["val_accuracy"] > 0.70, history[-1]["val_accuracy"]
