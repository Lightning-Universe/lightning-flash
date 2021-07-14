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
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Type, Union, Dict

import torch
from pytorch_lightning.metrics import Accuracy
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear

from flash.core.classification import ClassificationTask
from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.process import Serializer
from flash.core.registry import FlashRegistry
from flash.graph.backbones import GRAPH_CLASSIFICATION_BACKBONES
from flash.core.utilities.imports import _TORCH_GEOMETRIC_AVAILABLE

if _TORCH_GEOMETRIC_AVAILABLE:
    from torch_geometric.nn import BatchNorm, GCNConv, global_mean_pool, MessagePassing
else:
    MessagePassing = type(object)
    GCNConv = object


class GraphBlock(nn.Module):

    def __init__(self, nc_input, nc_output, conv_cls, act=nn.ReLU(), **conv_kwargs):
        super().__init__()
        self.conv = conv_cls(nc_input, nc_output, **conv_kwargs)
        self.norm = BatchNorm(nc_output)
        self.act = act

    def forward(self, x, edge_index, batch):
        x = self.conv(x, edge_index)
        x = self.norm(x)
        return self.act(x)


class BaseGraphModel(nn.Module):
    '''Deprecated in favour of GraphUNet'''

    def __init__(
        self,
        num_features: int,
        hidden_channels: List[int],
        num_classes: int,
        conv_cls: Type[MessagePassing],
        act=nn.ReLU(),
        **conv_kwargs: Any
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        hidden_channels = [num_features] + hidden_channels

        nc_output = num_features

        for idx in range(len(hidden_channels) - 1):
            nc_input = hidden_channels[idx]
            nc_output = hidden_channels[idx + 1]
            graph_block = GraphBlock(nc_input, nc_output, conv_cls, act, **conv_kwargs)
            self.blocks.append(graph_block)

        self.lin = Linear(nc_output, num_classes, act, **conv_kwargs)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        for block in self.blocks:
            x = block(x, edge_index, batch)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x


class GraphClassifier(ClassificationTask):
    """Task that classifies graphs.
    Some documentation https://1176-333857397-gh.circle-artifacts.com/0/html/custom_task.html

    Args:
        num_features: Number of columns in table (not including target column).
        num_classes: Number of classes to classify.
        hidden_channels: Hidden dimension sizes.
        loss_fn: Loss function for training, defaults to cross entropy.
        optimizer: Optimizer to use for training, defaults to `torch.optim.Adam`.
        metrics: Metrics to compute for training and evaluation.
        learning_rate: Learning rate to use for training, defaults to `1e-3`
        model: GraphNN used, defaults to BaseGraphModel.
        conv_cls: kind of convolution used in model, defaults to GCNConv
    """
    backbones: FlashRegistry = GRAPH_CLASSIFICATION_BACKBONES

    required_extras: str = "graph"

    def __init__(
        self,
        num_classes: int,
        backbone: Union[str, Tuple[nn.Module, int]] = "GraphUNet",
        backbone_kwargs: Optional[Dict] = {'depth': 4},
        head: Optional[Union[FunctionType, nn.Module]] = None,
        hidden_channels: Union[List[int], int] = 512,
        loss_fn: Callable = F.cross_entropy,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        metrics: Union[Callable, Mapping, Sequence, None] = [Accuracy()],
        learning_rate: float = 1e-3,
        conv_cls: Type[MessagePassing] = GCNConv,
        **conv_kwargs
    ):

        self.save_hyperparameters()

        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]

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
            self.backbone, num_out_features = self.backbones.get(backbone)(pretrained=False, **backbone_kwargs)

        head = head(num_out_features, num_classes) if isinstance(head, FunctionType) else head
        self.head = head or nn.Sequential(nn.Linear(num_out_features, num_classes), )

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
        batch[DefaultDataKeys.PREDS] = super().predict_step((batch[DefaultDataKeys.INPUT]),
                                                            batch_idx,
                                                            dataloader_idx=dataloader_idx)
        return batch

    def forward(self, x) -> torch.Tensor:
        x = self.backbone(x)
        return self.head(x)