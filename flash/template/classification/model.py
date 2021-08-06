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
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Type, Union

import torch
import torchmetrics
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler

from flash.core.classification import ClassificationTask, Labels
from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.process import Serializer
from flash.core.registry import FlashRegistry
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
        optimizer: The optimizer or optimizer class to use.
        optimizer_kwargs: Additional kwargs to use when creating the optimizer (if not passed as an instance).
        scheduler: The scheduler or scheduler class to use.
        scheduler_kwargs: Additional kwargs to use when creating the scheduler (if not passed as an instance).
        metrics: Any metrics to use with this :class:`~flash.core.model.Task`. If ``None``, a default will be selected
            by the :class:`~flash.core.classification.ClassificationTask` depending on the ``multi_label`` argument.
        learning_rate: The learning rate for the optimizer.
        multi_label: If ``True``, this will be treated as a multi-label classification problem.
        serializer: The :class:`~flash.core.data.process.Serializer` to use for prediction outputs.
    """

    backbones: FlashRegistry = TEMPLATE_BACKBONES

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        backbone: Union[str, Tuple[nn.Module, int]] = "mlp-128",
        backbone_kwargs: Optional[Dict] = None,
        loss_fn: Optional[Callable] = None,
        optimizer: Union[Type[torch.optim.Optimizer], torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Union[Type[_LRScheduler], str, _LRScheduler]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        metrics: Union[torchmetrics.Metric, Mapping, Sequence, None] = None,
        learning_rate: float = 1e-2,
        multi_label: bool = False,
        serializer: Optional[Union[Serializer, Mapping[str, Serializer]]] = None,
    ):
        super().__init__(
            model=None,
            loss_fn=loss_fn,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            metrics=metrics,
            learning_rate=learning_rate,
            multi_label=multi_label,
            serializer=serializer or Labels(),
        )

        self.save_hyperparameters()

        if not backbone_kwargs:
            backbone_kwargs = {}

        if isinstance(backbone, tuple):
            self.backbone, out_features = backbone
        else:
            self.backbone, out_features = self.backbones.get(backbone)(num_features=num_features, **backbone_kwargs)

        self.head = nn.Linear(out_features, num_classes)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        """For the training step, we just extract the :attr:`~flash.core.data.data_source.DefaultDataKeys.INPUT` and
        :attr:`~flash.core.data.data_source.DefaultDataKeys.TARGET` keys from the input and forward them to the
        :meth:`~flash.core.model.Task.training_step`."""
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        """For the validation step, we just extract the :attr:`~flash.core.data.data_source.DefaultDataKeys.INPUT` and
        :attr:`~flash.core.data.data_source.DefaultDataKeys.TARGET` keys from the input and forward them to the
        :meth:`~flash.core.model.Task.validation_step`."""
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        """For the test step, we just extract the :attr:`~flash.core.data.data_source.DefaultDataKeys.INPUT` and
        :attr:`~flash.core.data.data_source.DefaultDataKeys.TARGET` keys from the input and forward them to the
        :meth:`~flash.core.model.Task.test_step`."""
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return super().test_step(batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """For the predict step, we just extract the :attr:`~flash.core.data.data_source.DefaultDataKeys.INPUT` key
        from the input and forward it to the :meth:`~flash.core.model.Task.predict_step`."""
        batch = batch[DefaultDataKeys.INPUT]
        return super().predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)

    def forward(self, x) -> torch.Tensor:
        """First call the backbone, then the model head."""
        x = self.backbone(x)
        return self.head(x)
