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
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Type, Union

import torch
import torchmetrics
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler

from flash.core.classification import ClassificationTask
from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.process import Serializer


class TemplateSKLearnClassifier(ClassificationTask):
    """The ``TemplateSKLearnClassifier`` is a :class:`~flash.core.classification.ClassificationTask` that uses a simple
    multi-layer perceptron model to classify tabular data from scikit-learn. In the ``__init__``, we create our model
    and pass it to the super :class:`~flash.core.model.Task` along with any arguments that we need.

    Args:
        num_features: The number of features (elements) in the input data.
        num_classes: The number of classes (outputs) for this :class:`~flash.core.model.Task`.
        hidden_size: The number of units to use in the hidden layer of the multi-layer perceptron model.
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

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_size: 128,
        loss_fn: Optional[Callable] = None,
        optimizer: Union[Type[torch.optim.Optimizer], torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Union[Type[_LRScheduler], str, _LRScheduler]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        metrics: Union[torchmetrics.Metric, Mapping, Sequence, None] = None,
        learning_rate: float = 1e-3,
        multi_label: bool = False,
        serializer: Optional[Union[Serializer, Mapping[str, Serializer]]] = None,
    ):
        model = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, num_classes),
        )

        super().__init__(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            metrics=metrics,
            learning_rate=learning_rate,
            multi_label=multi_label,
            serializer=serializer,
        )

        self.save_hyperparameters()

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
        """For the predict step, we just extract the :attr:`~flash.core.data.data_source.DefaultDataKeys.INPUT` key from
        the input and forward it to the :meth:`~flash.core.model.Task.predict_step`."""
        batch = (batch[DefaultDataKeys.INPUT])
        return super().predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)
