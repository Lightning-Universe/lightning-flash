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
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import Metric

from flash.core.classification import ClassificationAdapterTask, Labels
from flash.core.data.process import Serializer
from flash.core.registry import FlashRegistry
from flash.image.classification.adapters import TRAINING_STRATEGIES
from flash.image.classification.backbones import IMAGE_CLASSIFIER_BACKBONES


class ImageClassifier(ClassificationAdapterTask):
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
        optimizer_kwargs: Additional kwargs to use when creating the optimizer (if not passed as an instance).
        scheduler: The scheduler or scheduler class to use.
        scheduler_kwargs: Additional kwargs to use when creating the scheduler (if not passed as an instance).
        metrics: Metrics to compute for training and evaluation. Can either be an metric from the `torchmetrics`
            package, a custom metric inheriting from `torchmetrics.Metric`, a callable function or a list/dict
            containing a combination of the aforementioned. In all cases, each metric needs to have the signature
            `metric(preds,target)` and return a single scalar tensor. Defaults to :class:`torchmetrics.Accuracy`.
        learning_rate: Learning rate to use for training, defaults to ``1e-3``.
        multi_label: Whether the targets are multi-label or not.
        serializer: The :class:`~flash.core.data.process.Serializer` to use when serializing prediction outputs.
    """

    backbones: FlashRegistry = IMAGE_CLASSIFIER_BACKBONES
    training_strategies: FlashRegistry = TRAINING_STRATEGIES

    required_extras: str = "image"

    def __init__(
        self,
        num_classes: Optional[int] = None,
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
        training_strategy: Optional[str] = "default",
        training_strategy_kwargs: Optional[Dict[str, Any]] = None,
    ):

        self.save_hyperparameters()

        if not backbone_kwargs:
            backbone_kwargs = {}

        if not training_strategy_kwargs:
            training_strategy_kwargs = {}

        if training_strategy == "default":
            if not num_classes:
                raise MisconfigurationException("`num_classes` should be provided.")
        else:
            num_classes = training_strategy_kwargs.get("ways", None)
            if not num_classes:
                raise MisconfigurationException(
                    "`training_strategy_kwargs` should contain `ways`, `meta_batch_size` and `shots`."
                )

        if isinstance(backbone, tuple):
            backbone, num_features = backbone
        else:
            backbone, num_features = self.backbones.get(backbone)(pretrained=pretrained, **backbone_kwargs)

        head = head(num_features, num_classes) if isinstance(head, FunctionType) else head
        head = head or nn.Sequential(
            nn.Linear(num_features, num_classes),
        )

        adapter_from_class = self.training_strategies.get(training_strategy)
        adapter = adapter_from_class(
            task=self,
            num_classes=num_classes,
            backbone=backbone,
            head=head,
            pretrained=pretrained,
            **training_strategy_kwargs,
        )

        super().__init__(
            adapter,
            num_classes=num_classes,
            loss_fn=loss_fn,
            metrics=metrics,
            learning_rate=learning_rate,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            multi_label=multi_label,
            serializer=serializer or Labels(multi_label=multi_label),
        )

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
