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
from typing import Any, Callable, Dict, List, Tuple

import torch
from omegaconf import OmegaConf
from pytorch_tabular.config import OptimizerConfig
from pytorch_tabular.models import TabNetModel, TabNetModelConfig
from torch.nn import functional as F

from flash.core.classification import ClassificationTask, Probabilities
from flash.core.data.io.input import DataKeys
from flash.core.integrations.pytorch_tabular.backbones import PYTORCH_TABULAR_BACKBONES
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _TABULAR_AVAILABLE
from flash.core.utilities.types import LR_SCHEDULER_TYPE, METRICS_TYPE, OPTIMIZER_TYPE, OUTPUT_TYPE

if _TABULAR_AVAILABLE:
    from pytorch_tabnet.tab_network import TabNet


# class TabularClassifier(AdapterTask):
#     backbones: FlashRegistry = FlashRegistry("backbones") + PYTORCH_FORECASTING_BACKBONES
#
#     def __init__(
#         self,
#         parameters: Dict[str, Any],
#         backbone: str,
#         backbone_kwargs: Optional[Dict[str, Any]] = None,
#         loss_fn: Optional[Callable] = None,
#         optimizer: OPTIMIZER_TYPE = "Adam",
#         lr_scheduler: LR_SCHEDULER_TYPE = None,
#         metrics: Union[torchmetrics.Metric, List[torchmetrics.Metric]] = None,
#         learning_rate: float = 4e-3,
#     ):
#
#         self.save_hyperparameters()
#
#         if backbone_kwargs is None:
#             backbone_kwargs = {}
#
#         metadata = self.backbones.get(backbone, with_metadata=True)
#         adapter = metadata["metadata"]["adapter"].from_task(
#             self,
#             parameters=parameters,
#             backbone=backbone,
#             backbone_kwargs=backbone_kwargs,
#             loss_fn=loss_fn,
#             metrics=metrics,
#         )
#
#         super().__init__(
#             adapter,
#             learning_rate=learning_rate,
#             optimizer=optimizer,
#             lr_scheduler=lr_scheduler,
#         )


class TabularClassifier(ClassificationTask):
    """The ``TabularClassifier`` is a :class:`~flash.Task` for classifying tabular data. For more details, see
    :ref:`tabular_classification`.

    Args:
        num_features: Number of columns in table (not including target column).
        num_classes: Number of classes to classify.
        embedding_sizes: List of (num_classes, emb_dim) to form categorical embeddings.
        loss_fn: Loss function for training, defaults to cross entropy.
        optimizer: Optimizer to use for training.
        lr_scheduler: The LR scheduler to use during training.
        metrics: Metrics to compute for training and evaluation. Can either be an metric from the `torchmetrics`
            package, a custom metric inherenting from `torchmetrics.Metric`, a callable function or a list/dict
            containing a combination of the aforementioned. In all cases, each metric needs to have the signature
            `metric(preds,target)` and return a single scalar tensor. Defaults to :class:`torchmetrics.Accuracy`.
        learning_rate: Learning rate to use for training.
        multi_label: Whether the targets are multi-label or not.
        output: The :class:`~flash.core.data.io.output.Output` to use when formatting prediction outputs.
        **tabnet_kwargs: Optional additional arguments for the TabNet model, see
            `pytorch_tabnet <https://dreamquark-ai.github.io/tabnet/_modules/pytorch_tabnet/tab_network.html#TabNet>`_.
    """

    required_extras: str = "tabular"
    backbones: FlashRegistry = FlashRegistry("backbones") + PYTORCH_TABULAR_BACKBONES

    def __init__(
        self,
        properties,
        backbone,
        embedding_sizes: List[Tuple[int, int]] = None,
        loss_fn: Callable = F.cross_entropy,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        metrics: METRICS_TYPE = None,
        learning_rate: float = 1e-2,
        multi_label: bool = False,
        output: OUTPUT_TYPE = None,
        **tabnet_kwargs,
    ):
        self.save_hyperparameters()
        properties.update(tabnet_kwargs)
        metadata = self.backbones.get(backbone, with_metadata=True)
        adapter = metadata["metadata"]["adapter"].from_task(
            self,
            task_type="classification",
            parameters=properties,
            backbone=backbone,
            backbone_kwargs=tabnet_kwargs,
            loss_fn=loss_fn,
            metrics=metrics,
        )

        super().__init__(
            adapter,
            learning_rate=learning_rate,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

    def forward(self, x_in) -> torch.Tensor:
        # TabNet takes single input, x_in is composed of (categorical, numerical)
        #xs = [x for x in x_in if x.numel()]
        #x = torch.cat(xs, dim=1)
        x = {
            "categorical": x_in[0],
            "continuous": x_in[1]
        }
        return self.model.backbone(x)["logits"]

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
        batch = batch[DataKeys.INPUT]
        return self(batch)

    @classmethod
    def from_data(cls, datamodule, **kwargs) -> "TabularClassifier":
        model = cls(datamodule.num_features, datamodule.num_classes, datamodule.embedding_sizes, **kwargs)
        return model

    @staticmethod
    def _ci_benchmark_fn(history: List[Dict[str, Any]]):
        """This function is used only for debugging usage with CI."""
        assert history[-1]["val_accuracy"] > 0.6, history[-1]["val_accuracy"]
