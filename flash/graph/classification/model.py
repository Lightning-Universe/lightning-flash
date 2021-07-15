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
from torch.nn import functional as F
from torch.nn import Linear

from flash.core.classification import ClassificationTask
from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.process import Serializer
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _GRAPH_AVAILABLE
from flash.graph.backbones import GRAPH_CLASSIFICATION_BACKBONES


class GraphClassifier(ClassificationTask):
    """The ``GraphClassifier`` is a :class:`~flash.Task` for classifying graphs. For more details, see
    :ref:`graph_classification`.

    Args:
        num_classes: Number of classes to classify.
        backbone_kwargs: Dictionary dependent on the backbone, containing for example in_channels, out_channels, hidden_channels or depth (number of layers).
        backbone: Name of the backbone to use.
        loss_fn: Loss function for training, defaults to cross entropy.
        optimizer: Optimizer to use for training, defaults to `torch.optim.Adam`.
        metrics: Metrics to compute for training and evaluation.
        learning_rate: Learning rate to use for training, defaults to `1e-3`
        model: GraphNN used, defaults to BaseGraphModel.
        conv_cls: kind of convolution used in model, defaults to GCNConv
    """
    backbones: FlashRegistry = GRAPH_CLASSIFICATION_BACKBONES

    required_extras: str = "graph"

    required_extras = "graph"

    def __init__(
        self,
        num_classes: int,
        backbone_kwargs: Optional[Dict],
        backbone: Union[str, Tuple[nn.Module, int]] = "GraphUNet",
        head: Optional[Union[FunctionType, nn.Module]] = None,
        loss_fn: Callable = F.cross_entropy,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        metrics: Union[Callable, Mapping, Sequence, None] = None,
        learning_rate: float = 1e-3,
    ):

        self.save_hyperparameters()

        super().__init__(
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,
        )

        self.save_hyperparameters()

        if isinstance(backbone, tuple):
            self.backbone, num_out_features = backbone
        else:
            self.backbone = self.backbones.get(backbone)(**backbone_kwargs)
            num_out_features = backbone.hidden_channels

        head = head(num_out_features, num_classes) if isinstance(head, FunctionType) else head
        self.head = head or default_head(num_out_features, num_classes)

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
        batch[DefaultDataKeys.PREDS] = super().predict_step((batch[DefaultDataKeys.INPUT]),
                                                            batch_idx,
                                                            dataloader_idx=dataloader_idx)
        return batch

    def forward(self, x) -> torch.Tensor:
        x = self.backbone(x)
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
