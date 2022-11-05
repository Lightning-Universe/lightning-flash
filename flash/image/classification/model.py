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
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from torch import nn

from flash.core.classification import ClassificationAdapterTask
from flash.core.data.io.input import ServeInput
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
from flash.image.classification.adapters import TRAINING_STRATEGIES
from flash.image.classification.backbones import IMAGE_CLASSIFIER_BACKBONES
from flash.image.classification.heads import IMAGE_CLASSIFIER_HEADS
from flash.image.classification.input_transform import ImageClassificationInputTransform
from flash.image.data import ImageDeserializer


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
        head: A string from ``ImageClassifier.available_heads()``, an ``nn.Module``, or a function of (``num_features``,
            ``num_classes``) which returns an ``nn.Module`` to use as the model head.
        pretrained: A bool or string to specify the pretrained weights of the backbone, defaults to ``True``
            which loads the default supervised pretrained weights.
        loss_fn: Loss function for training, defaults to :func:`torch.nn.functional.cross_entropy`.
        optimizer: Optimizer to use for training.
        lr_scheduler: The LR scheduler to use during training.
        metrics: Metrics to compute for training and evaluation. Can either be an metric from the `torchmetrics`
            package, a custom metric inheriting from `torchmetrics.Metric`, a callable function or a list/dict
            containing a combination of the aforementioned. In all cases, each metric needs to have the signature
            `metric(preds,target)` and return a single scalar tensor. Defaults to :class:`torchmetrics.Accuracy`.
        learning_rate: Learning rate to use for training, defaults to ``1e-3``.
        multi_label: Whether the targets are multi-label or not.
        training_strategy: string indicating the training strategy. Adjust if you want to use `learn2learn`
            for doing meta-learning research
        training_strategy_kwargs: Additional kwargs for setting the training strategy
    """

    backbones: FlashRegistry = IMAGE_CLASSIFIER_BACKBONES
    heads: FlashRegistry = IMAGE_CLASSIFIER_HEADS
    training_strategies: FlashRegistry = TRAINING_STRATEGIES
    required_extras: str = "image"

    def __init__(
        self,
        num_classes: Optional[int] = None,
        labels: Optional[List[str]] = None,
        backbone: Union[str, Tuple[nn.Module, int]] = "resnet18",
        backbone_kwargs: Optional[Dict] = None,
        head: Union[str, FunctionType, nn.Module] = "linear",
        pretrained: Union[bool, str] = True,
        loss_fn: LOSS_FN_TYPE = None,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        metrics: METRICS_TYPE = None,
        learning_rate: Optional[float] = None,
        multi_label: bool = False,
        training_strategy: Optional[str] = "default",
        training_strategy_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.save_hyperparameters()

        if labels is not None and num_classes is None:
            num_classes = len(labels)

        if not backbone_kwargs:
            backbone_kwargs = {}

        if not training_strategy_kwargs:
            training_strategy_kwargs = {}

        if training_strategy == "default":
            if num_classes is None and labels is None:
                raise TypeError("`num_classes` or `labels` should be provided.")
        else:
            num_classes = training_strategy_kwargs.get("ways", None)
            if not num_classes:
                raise TypeError("`training_strategy_kwargs` should contain `ways`, `meta_batch_size` and `shots`.")

        if isinstance(backbone, tuple):
            backbone, num_features = backbone
        else:
            backbone, num_features = self.backbones.get(backbone)(pretrained=pretrained, **backbone_kwargs)

        if isinstance(head, str):
            head = self.heads.get(head)(num_features=num_features, num_classes=num_classes)
        else:
            head = head(num_features, num_classes) if isinstance(head, FunctionType) else head

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
            lr_scheduler=lr_scheduler,
            multi_label=multi_label,
            labels=labels,
        )

    @classmethod
    def available_pretrained_weights(cls, backbone: str):
        result = cls.backbones.get(backbone, with_metadata=True)
        pretrained_weights = None

        if "weights_paths" in result["metadata"]:
            pretrained_weights = list(result["metadata"]["weights_paths"].keys())

        return pretrained_weights

    @requires("serve")
    def serve(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        sanity_check: bool = True,
        input_cls: Optional[Type[ServeInput]] = ImageDeserializer,
        transform: INPUT_TRANSFORM_TYPE = ImageClassificationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        output: Optional[Union[str, Output]] = None,
    ) -> Composition:
        return super().serve(host, port, sanity_check, input_cls, transform, transform_kwargs, output)

    def _ci_benchmark_fn(self, history: List[Dict[str, Any]]):
        """This function is used only for debugging usage with CI."""
        if self.hparams.multi_label:
            assert history[-1]["val_f1score"] > 0.30, history[-1]["val_f1score"]
        else:
            assert history[-1]["val_accuracy"] > 0.85, history[-1]["val_accuracy"]
