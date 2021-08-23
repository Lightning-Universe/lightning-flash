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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import torch
from torch.nn import functional as F
from torchmetrics import Metric

from flash.core.classification import ClassificationTask, Probabilities
from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.process import Serializer
from flash.core.utilities.imports import _TABULAR_AVAILABLE

if _TABULAR_AVAILABLE:
    from pytorch_tabnet.tab_network import TabNet


class TabularClassifier(ClassificationTask):
    """The ``TabularClassifier`` is a :class:`~flash.Task` for classifying tabular data. For more details, see
    :ref:`tabular_classification`.

    Args:
        num_features: Number of columns in table (not including target column).
        num_classes: Number of classes to classify.
        embedding_sizes: List of (num_classes, emb_dim) to form categorical embeddings.
        loss_fn: Loss function for training, defaults to cross entropy.
        optimizer: Optimizer to use for training, defaults to `torch.optim.Adam`.
        metrics: Metrics to compute for training and evaluation. Can either be an metric from the `torchmetrics`
            package, a custom metric inherenting from `torchmetrics.Metric`, a callable function or a list/dict
            containing a combination of the aforementioned. In all cases, each metric needs to have the signature
            `metric(preds,target)` and return a single scalar tensor. Defaults to :class:`torchmetrics.Accuracy`.
        learning_rate: Learning rate to use for training, defaults to `1e-3`
        multi_label: Whether the targets are multi-label or not.
        serializer: The :class:`~flash.core.data.process.Serializer` to use when serializing prediction outputs.
        **tabnet_kwargs: Optional additional arguments for the TabNet model, see
            `pytorch_tabnet <https://dreamquark-ai.github.io/tabnet/_modules/pytorch_tabnet/tab_network.html#TabNet>`_.
    """

    required_extras: str = "tabular"

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        embedding_sizes: List[Tuple[int, int]] = None,
        loss_fn: Callable = F.cross_entropy,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        metrics: Union[Metric, Callable, Mapping, Sequence, None] = None,
        learning_rate: float = 1e-2,
        multi_label: bool = False,
        serializer: Optional[Union[Serializer, Mapping[str, Serializer]]] = None,
        **tabnet_kwargs,
    ):
        self.save_hyperparameters()

        cat_dims, cat_emb_dim = zip(*embedding_sizes) if embedding_sizes else ([], [])
        model = TabNet(
            input_dim=num_features,
            output_dim=num_classes,
            cat_idxs=list(range(len(embedding_sizes))),
            cat_dims=list(cat_dims),
            cat_emb_dim=list(cat_emb_dim),
            **tabnet_kwargs,
        )

        super().__init__(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,
            multi_label=multi_label,
            serializer=serializer or Probabilities(),
        )

        self.save_hyperparameters()

    def forward(self, x_in) -> torch.Tensor:
        # TabNet takes single input, x_in is composed of (categorical, numerical)
        xs = []
        for x in x_in:
            if x.numel():
                xs.append(x)
        x = torch.cat(xs, dim=1)
        return self.model(x)[0]

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
        batch = batch[DefaultDataKeys.INPUT]
        return self(batch)

    @classmethod
    def from_data(cls, datamodule, **kwargs) -> "TabularClassifier":
        model = cls(datamodule.num_features, datamodule.num_classes, datamodule.embedding_sizes, **kwargs)
        return model

    @staticmethod
    def _ci_benchmark_fn(history: List[Dict[str, Any]]):
        """This function is used only for debugging usage with CI."""
        assert history[-1]["val_accuracy"] > 0.6, history[-1]["val_accuracy"]
