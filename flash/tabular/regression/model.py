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
from typing import Any, Callable, List, Tuple

import torch
from torch.nn import functional as F

from flash.core.data.data_source import DefaultDataKeys
from flash.core.regression import RegressionTask
from flash.core.utilities.imports import _TABULAR_AVAILABLE
from flash.core.utilities.types import LR_SCHEDULER_TYPE, METRICS_TYPE, OPTIMIZER_TYPE, SERIALIZER_TYPE

if _TABULAR_AVAILABLE:
    from pytorch_tabnet.tab_network import TabNet


class TabularRegressor(RegressionTask):
    """The ``TabularRegressor`` is a :class:`~flash.Task` for regression tabular data.

    Args:
        num_features: Number of columns in table (not including target column).
        embedding_sizes: List of (num_classes, emb_dim) to form categorical embeddings.
        loss_fn: Loss function for training, defaults to cross entropy.
        optimizer: Optimizer to use for training.
        lr_scheduler: The LR scheduler to use during training.
        metrics: Metrics to compute for training and evaluation. Can either be an metric from the `torchmetrics`
            package, a custom metric inherenting from `torchmetrics.Metric`, a callable function or a list/dict
            containing a combination of the aforementioned. In all cases, each metric needs to have the signature
            `metric(preds,target)` and return a single scalar tensor.
        learning_rate: Learning rate to use for training.
        multi_label: Whether the targets are multi-label or not.
        serializer: The :class:`~flash.core.data.process.Serializer` to use when serializing prediction outputs.
        **tabnet_kwargs: Optional additional arguments for the TabNet model, see
            `pytorch_tabnet <https://dreamquark-ai.github.io/tabnet/_modules/pytorch_tabnet/tab_network.html#TabNet>`_.
    """

    required_extras: str = "tabular"

    def __init__(
        self,
        num_features: int,
        embedding_sizes: List[Tuple[int, int]] = None,
        loss_fn: Callable = F.mse_loss,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        metrics: METRICS_TYPE = None,
        learning_rate: float = 1e-2,
        serializer: SERIALIZER_TYPE = None,
        **tabnet_kwargs,
    ):
        self.save_hyperparameters()

        cat_dims, cat_emb_dim = zip(*embedding_sizes) if embedding_sizes else ([], [])
        model = TabNet(
            input_dim=num_features,
            output_dim=1,
            cat_idxs=list(range(len(embedding_sizes))),
            cat_dims=list(cat_dims),
            cat_emb_dim=list(cat_emb_dim),
            **tabnet_kwargs,
        )

        super().__init__(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            learning_rate=learning_rate,
            serializer=serializer,
        )

        self.save_hyperparameters()

    def forward(self, x_in) -> torch.Tensor:
        # TabNet takes single input, x_in is composed of (categorical, numerical)
        xs = [x for x in x_in if x.numel()]
        x = torch.cat(xs, dim=1)
        return self.model(x)[0].flatten()

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
    def from_data(cls, datamodule, **kwargs) -> "TabularRegressor":
        model = cls(datamodule.num_features, datamodule.embedding_sizes, **kwargs)
        return model
