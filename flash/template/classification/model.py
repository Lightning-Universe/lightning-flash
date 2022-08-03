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
from typing import Any, Dict, List, Optional, Tuple, Union

from torch import nn, Tensor

from flash.core.classification import ClassificationTask
from flash.core.data.io.input import DataKeys
from flash.core.registry import FlashRegistry
from flash.core.utilities.types import LOSS_FN_TYPE, LR_SCHEDULER_TYPE, METRICS_TYPE, OPTIMIZER_TYPE
from flash.template.classification.backbones import TEMPLATE_BACKBONES


class TemplateSKLearnClassifier(ClassificationTask):
    """The ``TemplateSKLearnClassifier`` is a :class:`~flash.core.classification.ClassificationTask` that
    classifies tabular data from scikit-learn.

    Args:
        num_features: The number of features (elements) in the input data.
        num_classes: The number of classes (outputs) for this :class:`~flash.core.model.Task`.
        backbone: The backbone name (or a tuple of ``nn.Module``, output size) to use.
        backbone_kwargs: Any additional kwargs to pass to the backbone constructor.
        loss_fn: The loss function to use. If ``None``, a default will be selected by the
            :class:`~flash.core.classification.ClassificationTask` depending on the ``multi_label`` argument.
        optimizer: Optimizer to use for training.
        lr_scheduler: The LR scheduler to use during training.
        metrics: Any metrics to use with this :class:`~flash.core.model.Task`. If ``None``, a default will be selected
            by the :class:`~flash.core.classification.ClassificationTask` depending on the ``multi_label`` argument.
        learning_rate: The learning rate for the optimizer.
        multi_label: If ``True``, this will be treated as a multi-label classification problem.
    """

    backbones: FlashRegistry = TEMPLATE_BACKBONES

    def __init__(
        self,
        num_features: int,
        num_classes: Optional[int] = None,
        labels: Optional[List[str]] = None,
        backbone: Union[str, Tuple[nn.Module, int]] = "mlp-128",
        backbone_kwargs: Optional[Dict] = None,
        loss_fn: LOSS_FN_TYPE = None,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        metrics: METRICS_TYPE = None,
        learning_rate: Optional[float] = None,
        multi_label: bool = False,
    ):
        self.save_hyperparameters()

        if labels is not None and num_classes is None:
            num_classes = len(labels)

        super().__init__(
            model=None,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            learning_rate=learning_rate,
            multi_label=multi_label,
            num_classes=num_classes,
            labels=labels,
        )

        if not backbone_kwargs:
            backbone_kwargs = {}

        if isinstance(backbone, tuple):
            self.backbone, out_features = backbone
        else:
            self.backbone, out_features = self.backbones.get(backbone)(num_features=num_features, **backbone_kwargs)

        self.head = nn.Linear(out_features, num_classes)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        """For the training step, we just extract the :attr:`~flash.core.data.io.input.DataKeys.INPUT` and
        :attr:`~flash.core.data.io.input.DataKeys.TARGET` keys from the input and forward them to the
        :meth:`~flash.core.model.Task.training_step`."""
        batch = (batch[DataKeys.INPUT], batch[DataKeys.TARGET])
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        """For the validation step, we just extract the :attr:`~flash.core.data.io.input.DataKeys.INPUT` and
        :attr:`~flash.core.data.io.input.DataKeys.TARGET` keys from the input and forward them to the
        :meth:`~flash.core.model.Task.validation_step`."""
        batch = (batch[DataKeys.INPUT], batch[DataKeys.TARGET])
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        """For the test step, we just extract the :attr:`~flash.core.data.io.input.DataKeys.INPUT` and
        :attr:`~flash.core.data.io.input.DataKeys.TARGET` keys from the input and forward them to the
        :meth:`~flash.core.model.Task.test_step`."""
        batch = (batch[DataKeys.INPUT], batch[DataKeys.TARGET])
        return super().test_step(batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """For the predict step, we just extract the :attr:`~flash.core.data.io.input.DataKeys.INPUT` key from the
        input and forward it to the :meth:`~flash.core.model.Task.predict_step`."""
        batch = batch[DataKeys.INPUT]
        return super().predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)

    def forward(self, x) -> Tensor:
        """First call the backbone, then the model head."""
        x = self.backbone(x)
        return self.head(x)
