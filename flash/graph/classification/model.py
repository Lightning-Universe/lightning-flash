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
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Type, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear
from torch.optim.lr_scheduler import _LRScheduler

from flash.core.classification import ClassificationTask
from flash.core.data.data_source import DefaultDataKeys
from flash.core.registry import FlashRegistry
from flash.graph.backbones import GRAPH_BACKBONES


class GraphClassifier(ClassificationTask):
    """The ``GraphClassifier`` is a :class:`~flash.Task` for classifying graphs. For more details, see
    :ref:`graph_classification`.

    Args:
        num_features (int): The number of features in the input.
        num_classes (int): Number of classes to classify.
        backbone: Name of the backbone to use, defaults to ``"GCN"``.
        backbone_kwargs: Dictionary dependent on the backbone, containing for example in_channels, out_channels,
            hidden_channels or depth (number of layers).
        head: The head to use.
        loss_fn: Loss function for training, defaults to cross entropy.
        optimizer: Optimizer to use for training, defaults to `torch.optim.Adam`.
        optimizer_kwargs: Additional kwargs to use when creating the optimizer (if not passed as an instance).
        scheduler: The scheduler or scheduler class to use.
        scheduler_kwargs: Additional kwargs to use when creating the scheduler (if not passed as an instance).
        metrics: Metrics to compute for training and evaluation.
        learning_rate: Learning rate to use for training, defaults to `1e-3`
        model: GraphNN used, defaults to BaseGraphModel.
        conv_cls: kind of convolution used in model, defaults to GCNConv
    """

    backbones: FlashRegistry = GRAPH_BACKBONES

    required_extras: str = "graph"

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        backbone: Union[str, Tuple[nn.Module, int]] = "GCN",
        backbone_kwargs: Optional[Dict] = {},
        head: Optional[Union[FunctionType, nn.Module]] = None,
        loss_fn: Callable = F.cross_entropy,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Union[Type[_LRScheduler], str, _LRScheduler]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        metrics: Union[Callable, Mapping, Sequence, None] = None,
        learning_rate: float = 1e-3,
    ):

        self.save_hyperparameters()

        super().__init__(
            loss_fn=loss_fn,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            metrics=metrics,
            learning_rate=learning_rate,
        )

        self.save_hyperparameters()

        if isinstance(backbone, tuple):
            self.backbone, num_out_features = backbone
        else:
            self.backbone = self.backbones.get(backbone)(in_channels=num_features, **backbone_kwargs)
            num_out_features = self.backbone.hidden_channels

        if head is not None:
            self.head = head
        else:
            self.head = default_head(num_out_features, num_classes)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch, batch.y)
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch, batch.y)
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch, batch.y)
        return super().test_step(batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        batch[DefaultDataKeys.PREDS] = super().predict_step(
            (batch[DefaultDataKeys.INPUT]), batch_idx, dataloader_idx=dataloader_idx
        )
        return batch

    def forward(self, x) -> torch.Tensor:
        x = self.backbone(x.x, x.edge_index)
        return self.head(x)


class default_head(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes, dropout=0.5):
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)
