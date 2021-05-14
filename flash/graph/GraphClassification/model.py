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
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Type, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics import Accuracy
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool

from flash.core.classification import ClassificationTask
from flash.core.data import DataPipeline


class GraphClassifier(ClassificationTask):
    """Task that classifies graphs.

    Args:
            num_features: Number of columns in table (not including target column).
        num_classes: Number of classes to classify.
            embedding_sizes: List of (num_classes, emb_dim) to form categorical embeddings.
        hidden: Hidden dimension sizes.
        loss_fn: Loss function for training, defaults to cross entropy.
        optimizer: Optimizer to use for training, defaults to `torch.optim.Adam`.
        metrics: Metrics to compute for training and evaluation.
        learning_rate: Learning rate to use for training, defaults to `1e-3`
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden: Union[List[int], int] = 512,
        loss_fn: Callable = F.cross_entropy,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        metrics: Union[Callable, Mapping, Sequence, None] = [Accuracy()],
        learning_rate: float = 1e-3,
        model: torch.nn.Module = None,
    ):

        if isinstance(hidden, int):
            hidden = [hidden]

        #sizes = [input_size] + hidden + [num_classes]
        if model == None:
            self.model = GCN(in_features=num_features, hidden_channels=hidden, out_features=num_classes)

        super().__init__(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,
        )

    def forward(self, data) -> Any:
        x = self.model(data.x, data.edge_index, data.batch)
        return self.head(x)

    @staticmethod
    def default_pipeline() -> ClassificationDataPipeline:
        return GraphClassificationData.default_pipeline()


#Taken from https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing#scrollTo=CN3sRVuaQ88l
class GCN(pl.LightningModule):

    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x

    def training_step(self, batch, batch_idx):  #todo: is this needed?
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):  #todo: is this needed?
        return torch.optim.Adam(self.parameters(), lr=0.02)
