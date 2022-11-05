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
from torch import nn, Tensor

from flash.core.data.io.input import DataKeys
from flash.core.model import Task
from flash.graph.classification.model import GraphClassifier, POOLING_FUNCTIONS
from flash.graph.collate import _pyg_collate


class GraphEmbedder(Task):
    """The ``GraphEmbedder`` is a :class:`~flash.Task` for obtaining feature vectors (embeddings) from graphs. For
    more details, see :ref:`graph_embedder`.

    Args:
        backbone: A model to use to extract image features.
        pooling_fn: The global pooling operation to use (one of: "max", "max", "add" or a callable).
    """

    required_extras: str = "graph"

    def __init__(self, backbone: nn.Module, pooling_fn: Optional[Union[str, Callable]] = "mean"):
        super().__init__(model=None)

        # Don't save backbone or pooling_fn if it is not a string
        self.save_hyperparameters(ignore=["backbone"] if isinstance(pooling_fn, str) else ["backbone", "pooling_fn"])

        self.backbone = backbone

        self.pooling_fn = POOLING_FUNCTIONS[pooling_fn] if isinstance(pooling_fn, str) else pooling_fn

        self.collate_fn = _pyg_collate

    def forward(self, data) -> Tensor:
        x = self.backbone(data.x, data.edge_index)
        x = self.pooling_fn(x, data.batch)
        return x

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        raise NotImplementedError("Training a `GraphEmbedder` is not supported. Use a `GraphClassifier` instead.")

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        raise NotImplementedError("Validating a `GraphEmbedder` is not supported. Use a `GraphClassifier` instead.")

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        raise NotImplementedError("Testing a `GraphEmbedder` is not supported. Use a `GraphClassifier` instead.")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return super().predict_step(batch[DataKeys.INPUT], batch_idx, dataloader_idx=dataloader_idx)

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

        return cls(classifier.backbone, classifier.pooling_fn)
