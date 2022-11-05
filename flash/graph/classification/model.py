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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn import Linear

from flash.core.classification import ClassificationTask
from flash.core.data.io.input import DataKeys
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _GRAPH_AVAILABLE
from flash.core.utilities.types import LOSS_FN_TYPE, LR_SCHEDULER_TYPE, METRICS_TYPE, OPTIMIZER_TYPE
from flash.graph.backbones import GRAPH_BACKBONES
from flash.graph.collate import _pyg_collate

if _GRAPH_AVAILABLE:
    from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

    POOLING_FUNCTIONS = {"mean": global_mean_pool, "add": global_add_pool, "max": global_max_pool}
else:
    POOLING_FUNCTIONS = {}


class GraphClassifier(ClassificationTask):
    """The ``GraphClassifier`` is a :class:`~flash.Task` for classifying graphs. For more details, see
    :ref:`graph_classification`.

    Args:
        num_features (int): The number of features in the input.
        num_classes (int): Number of classes to classify.
        backbone: Name of the backbone to use.
        backbone_kwargs: Dictionary dependent on the backbone, containing for example in_channels, out_channels,
            hidden_channels or depth (number of layers).
        pooling_fn: The global pooling operation to use (one of: "max", "max", "add" or a callable).
        head: The head to use.
        loss_fn: Loss function for training, defaults to cross entropy.
        learning_rate: Learning rate to use for training.
        optimizer: Optimizer to use for training.
        lr_scheduler: The LR scheduler to use during training.
        metrics: Metrics to compute for training and evaluation.
    """

    backbones: FlashRegistry = GRAPH_BACKBONES

    required_extras: str = "graph"

    def __init__(
        self,
        num_features: int,
        num_classes: Optional[int] = None,
        labels: Optional[List[str]] = None,
        backbone: Union[str, Tuple[nn.Module, int]] = "GCN",
        backbone_kwargs: Optional[Dict] = {},
        pooling_fn: Optional[Union[str, Callable]] = "mean",
        head: Optional[Union[Callable, nn.Module]] = None,
        loss_fn: LOSS_FN_TYPE = F.cross_entropy,
        learning_rate: Optional[float] = None,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        metrics: METRICS_TYPE = None,
    ):
        self.save_hyperparameters()

        if labels is not None and num_classes is None:
            num_classes = len(labels)

        super().__init__(
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            learning_rate=learning_rate,
            num_classes=num_classes,
            labels=labels,
        )

        self.save_hyperparameters()

        if isinstance(backbone, tuple):
            self.backbone, num_out_features = backbone
        else:
            self.backbone = self.backbones.get(backbone)(in_channels=num_features, **backbone_kwargs)
            num_out_features = self.backbone.hidden_channels

        self.pooling_fn = POOLING_FUNCTIONS[pooling_fn] if isinstance(pooling_fn, str) else pooling_fn

        if head is not None:
            self.head = head
        else:
            self.head = DefaultGraphHead(num_out_features, num_classes)

        self.collate_fn = _pyg_collate

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DataKeys.INPUT], batch[DataKeys.TARGET])
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DataKeys.INPUT], batch[DataKeys.TARGET])
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DataKeys.INPUT], batch[DataKeys.TARGET])
        return super().test_step(batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return super().predict_step(batch[DataKeys.INPUT], batch_idx, dataloader_idx=dataloader_idx)

    def forward(self, data) -> Tensor:
        x = self.backbone(data.x, data.edge_index)
        x = self.pooling_fn(x, data.batch)
        return self.head(x)


class DefaultGraphHead(nn.Module):
    def __init__(self, hidden_channels, num_classes, dropout=0.5):
        super().__init__()
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)
