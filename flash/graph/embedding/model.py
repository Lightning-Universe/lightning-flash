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
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy, Metric

from flash.core.data.data_source import DefaultDataKeys
from flash.core.model import Task
from flash.core.registry import FlashRegistry
from flash.graph.backbones import GRAPH_BACKBONES
from flash.graph.classification.data import GraphClassificationPreprocess


class GraphEmbedder(Task):
    """The ``GraphEmbedder`` is a :class:`~flash.Task` for obtaining feature vectors (embeddings) from graphs. For
    more details, see :ref:`graph_embedder`.

    Args:
        num_features (int): The number of features in the input.
        embedding_dim (int): Dimension of the embedded vector. ``None`` uses the default from the backbone.
        backbone: A model to use to extract image features, defaults to ``"GCN"``.
        backbone_kwargs (dict): Keyword arguments to pass to the backbone constructor.
        pretrained: Use a pretrained backbone, defaults to ``True``.
        loss_fn: Loss function for training and finetuning, defaults to :func:`torch.nn.functional.cross_entropy`
        optimizer: Optimizer to use for training and finetuning, defaults to :class:`torch.optim.SGD`.
        metrics: Metrics to compute for training and evaluation. Can either be an metric from the `torchmetrics`
            package, a custom metric inherenting from `torchmetrics.Metric`, a callable function or a list/dict
            containing a combination of the aforementioned. In all cases, each metric needs to have the signature
            `metric(preds,target)` and return a single scalar tensor. Defaults to :class:`torchmetrics.Accuracy`.
        learning_rate: Learning rate to use for training, defaults to ``1e-3``.
        pooling_fn: Function used to pool image to generate embeddings, defaults to :func:`torch.max`.
    """

    backbones: FlashRegistry = GRAPH_BACKBONES

    required_extras: str = "graph"

    def __init__(
        self,
        num_features: int,
        embedding_dimension: Optional[int] = None,
        backbone: Union[str, Tuple[nn.Module, int]] = "GCN",
        backbone_kwargs: Optional[Dict] = {},
        pretrained: Optional[bool] = None,
        loss_fn: Callable = F.cross_entropy,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.SGD,
        metrics: Union[Metric, Callable, Mapping, Sequence, None] = (Accuracy()),
        learning_rate: float = 1e-3,
        pooling_fn: Callable = torch.max,
    ):
        super().__init__(
            model=None,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,
            preprocess=GraphClassificationPreprocess(),
        )

        self.save_hyperparameters()
        self.backbone_name = backbone
        self.embedding_dimension = embedding_dimension
        assert pooling_fn in [torch.mean, torch.max]
        self.pooling_fn = pooling_fn

        self.backbone = self.backbones.get(backbone)(in_channels=num_features, pretrained=pretrained, **backbone_kwargs)
        num_out_features = backbone.hidden_channels
        if self.embedding_dimension is not None:
            self.head = nn.Sequential(nn.Linear(num_out_features, self.embedding_dimension))
        else:
            self.head = nn.Sequential(nn.Linear(num_out_features, num_out_features))

    def forward(self, x) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x

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
        batch = batch[DefaultDataKeys.INPUT]
        return super().predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)

    @classmethod
    def load_from_checkpoint(self, path: str, strict: bool = True) -> None:
        checkpoint = torch.load(checkpoint)

        self.backbone.load_state_dict(checkpoint["backbone"], strict=strict)
        # self.head.load_state_dict(state_dict["head"], strict=strict)

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.learning_rate = checkpoint["learning_rate"]
        self.loss_fn = checkpoint["loss_fn"]
        self.metrics = checkpoint["metrics"]
