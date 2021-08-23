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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import Metric

from flash.core.classification import ClassificationTask, Labels
from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.process import Serializer
from flash.core.registry import FlashRegistry
from flash.image.classification.backbones import IMAGE_CLASSIFIER_BACKBONES


class ImageClassifier(ClassificationTask):
    """The ``ImageClassifier`` is a :class:`~flash.Task` for classifying images. For more details, see
    :ref:`image_classification`. The ``ImageClassifier`` also supports multi-label classification with
    ``multi_label=True``. For more details, see :ref:`image_classification_multi_label`.

    You can register custom backbones to use with the ``ImageClassifier``:
    ::

        from torch import nn
        import torchvision
        from flash.image import ImageClassifier

        # This is useful to create new backbone and make them accessible from `ImageClassifier`
        @ImageClassifier.backbones(name="resnet18")
        def fn_resnet(pretrained: bool = True):
            model = torchvision.models.resnet18(pretrained)
            # remove the last two layers & turn it into a Sequential model
            backbone = nn.Sequential(*list(model.children())[:-2])
            num_features = model.fc.in_features
            # backbones need to return the num_features to build the head
            return backbone, num_features

    Args:
        num_classes: Number of classes to classify.
        backbone: A string or (model, num_features) tuple to use to compute image features, defaults to ``"resnet18"``.
        pretrained: A bool or string to specify the pretrained weights of the backbone, defaults to ``True``
            which loads the default supervised pretrained weights.
        loss_fn: Loss function for training, defaults to :func:`torch.nn.functional.cross_entropy`.
        optimizer: Optimizer to use for training, defaults to :class:`torch.optim.SGD`.
        metrics: Metrics to compute for training and evaluation. Can either be an metric from the `torchmetrics`
            package, a custom metric inheriting from `torchmetrics.Metric`, a callable function or a list/dict
            containing a combination of the aforementioned. In all cases, each metric needs to have the signature
            `metric(preds,target)` and return a single scalar tensor. Defaults to :class:`torchmetrics.Accuracy`.
        learning_rate: Learning rate to use for training, defaults to ``1e-3``.
        multi_label: Whether the targets are multi-label or not.
        serializer: The :class:`~flash.core.data.process.Serializer` to use when serializing prediction outputs.
    """

    backbones: FlashRegistry = IMAGE_CLASSIFIER_BACKBONES

    required_extras: str = "image"

    def __init__(
        self,
        num_classes: int,
        backbone: Union[str, Tuple[nn.Module, int]] = "resnet18",
        backbone_kwargs: Optional[Dict] = None,
        head: Optional[Union[FunctionType, nn.Module]] = None,
        pretrained: Union[bool, str] = True,
        loss_fn: Optional[Callable] = None,
        optimizer: Union[Type[torch.optim.Optimizer], torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Union[Type[_LRScheduler], str, _LRScheduler]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        metrics: Union[Metric, Callable, Mapping, Sequence, None] = None,
        learning_rate: float = 1e-3,
        multi_label: bool = False,
        serializer: Optional[Union[Serializer, Mapping[str, Serializer]]] = None,
    ):
        super().__init__(
            num_classes=num_classes,
            model=None,
            loss_fn=loss_fn,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            metrics=metrics,
            learning_rate=learning_rate,
            multi_label=multi_label,
            serializer=serializer or Labels(multi_label=multi_label),
        )

        self.save_hyperparameters()

        if not backbone_kwargs:
            backbone_kwargs = {}

        if isinstance(backbone, tuple):
            self.backbone, num_features = backbone
        else:
            self.backbone, num_features = self.backbones.get(backbone)(pretrained=pretrained, **backbone_kwargs)

        head = head(num_features, num_classes) if isinstance(head, FunctionType) else head
        self.head = head or nn.Sequential(
            nn.Linear(num_features, num_classes),
        )

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return super().test_step(batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        batch[DefaultDataKeys.PREDS] = super().predict_step(
            (batch[DefaultDataKeys.INPUT]), batch_idx, dataloader_idx=dataloader_idx
        )
        return batch

    def forward(self, x) -> torch.Tensor:
        x = self.backbone(x)
        if x.dim() == 4:
            x = x.mean(-1).mean(-1)
        return self.head(x)

    @classmethod
    def available_pretrained_weights(cls, backbone: str):
        result = cls.backbones.get(backbone, with_metadata=True)
        pretrained_weights = None

        if "weights_paths" in result["metadata"]:
            pretrained_weights = list(result["metadata"]["weights_paths"].keys())

        return pretrained_weights

    def _ci_benchmark_fn(self, history: List[Dict[str, Any]]):
        """This function is used only for debugging usage with CI."""
        if self.hparams.multi_label:
            assert history[-1]["val_f1"] > 0.40, history[-1]["val_f1"]
        else:
            assert history[-1]["val_accuracy"] > 0.85, history[-1]["val_accuracy"]
