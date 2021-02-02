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
from typing import Any, Callable, List, Optional, Tuple, Type, Union, Mapping, Sequence, Union

import torch
from pytorch_lightning.metrics import Accuracy
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

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
        num_classes: int,
        hidden: Union[List[int], int] = 512,
        loss_fn: Callable = F.cross_entropy,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        metrics: Union[Callable, Mapping, Sequence, None] = [Accuracy()],
        learning_rate: float = 1e-3,
    ):

        if isinstance(hidden, int):
            hidden = [hidden]

        sizes = [input_size] + hidden + [num_classes]

        super().__init__(
            model=self._init_model(sizes),
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,
        )

    def _init_model():
        return 

    def forward(self, x) -> Any:
        x = x = self.model(x)
        return self.head(x)

    @staticmethod
    def default_pipeline() -> ClassificationDataPipeline:
        return GraphClassificationData.default_pipeline()
