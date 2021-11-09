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
from typing import Any, Callable, Dict, IO, Optional, Union

import torch
from torch import nn

from flash.core.model import Task
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _GRAPH_AVAILABLE
from flash.graph.backbones import GRAPH_BACKBONES
from flash.graph.classification.data import GraphClassificationPreprocess
from flash.graph.classification.model import GraphClassifier

if _GRAPH_AVAILABLE:
    from torch_geometric.nn import global_mean_pool


class GraphEmbedder(Task):
    """The ``GraphEmbedder`` is a :class:`~flash.Task` for obtaining feature vectors (embeddings) from graphs. For
    more details, see :ref:`graph_embedder`.

    Args:
        backbone: A model to use to extract image features.
    """

    backbones: FlashRegistry = GRAPH_BACKBONES

    required_extras: str = "graph"

    def __init__(
        self,
        backbone: nn.Module,
    ):
        super().__init__(
            model=None,
            preprocess=GraphClassificationPreprocess(),
        )

        self.save_hyperparameters()

        self.backbone = backbone

    def forward(self, data) -> torch.Tensor:
        x = self.backbone(data.x, data.edge_index)
        x = global_mean_pool(x, data.batch)
        return x

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        raise NotImplementedError("Training a `GraphEmbedder` is not supported. Use a `GraphClassifier` instead.")

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        raise NotImplementedError("Validating a `GraphEmbedder` is not supported. Use a `GraphClassifier` instead.")

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        raise NotImplementedError("Testing a `GraphEmbedder` is not supported. Use a `GraphClassifier` instead.")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return super().predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, IO],
        map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
        hparams_file: Optional[str] = None,
        strict: bool = True,
        **kwargs,
    ) -> "GraphEmbedder":
        classifier = GraphClassifier.load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            hparams_file=hparams_file,
            strict=strict,
            **kwargs,
        )

        return cls(classifier.backbone)
