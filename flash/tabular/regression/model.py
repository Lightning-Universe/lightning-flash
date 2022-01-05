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
from functools import partial
from typing import Any, Callable, Dict, Optional, Type

from torch.nn import functional as F

from flash import InputTransform
from flash.core.data.io.input import ServeInput
from flash.core.integrations.pytorch_tabular.backbones import PYTORCH_TABULAR_BACKBONES
from flash.core.registry import FlashRegistry
from flash.core.regression import RegressionAdapterTask
from flash.core.serve import Composition
from flash.core.utilities.imports import requires
from flash.core.utilities.types import (
    INPUT_TRANSFORM_TYPE,
    LR_SCHEDULER_TYPE,
    METRICS_TYPE,
    OPTIMIZER_TYPE,
    OUTPUT_TYPE,
)
from flash.tabular.input import TabularDeserializer


class TabularRegressor(RegressionAdapterTask):
    """The ``TabularRegressor`` is a :class:`~flash.Task` for classifying tabular data. For more details, see
    :ref:`tabular_classification`.

    Args:
        embedding_sizes: Number of columns in table (not including target column).
        categorical_fields: Number of classes to classify.
        embedding_sizes: List of (num_classes, emb_dim) to form categorical embeddings.
        cat_dims: Number of distinct values for each categorical column
        num_categorical_fields: Number of categorical columns in table
        num_numerical_fields: Number of numerical columns in table
        output_dim: Number of output values
        backbone: name of the model to use
        loss_fn: Loss function for training, defaults to cross entropy.
        optimizer: Optimizer to use for training.
        lr_scheduler: The LR scheduler to use during training.
        metrics: Metrics to compute for training and evaluation. Can either be an metric from the `torchmetrics`
            package, a custom metric inherenting from `torchmetrics.Metric`, a callable function or a list/dict
            containing a combination of the aforementioned. In all cases, each metric needs to have the signature
            `metric(preds,target)` and return a single scalar tensor. Defaults to :class:`torchmetrics.Accuracy`.
        learning_rate: Learning rate to use for training.
        output: The :class:`~flash.core.data.io.output.Output` to use when formatting prediction outputs.
        **tabnet_kwargs: Optional additional arguments for the TabNet model, see
            `pytorch_tabnet <https://dreamquark-ai.github.io/tabnet/_modules/pytorch_tabnet/tab_network.html#TabNet>`_.
    """

    required_extras: str = "tabular"
    backbones: FlashRegistry = FlashRegistry("backbones") + PYTORCH_TABULAR_BACKBONES

    def __init__(
        self,
        embedding_sizes: list,
        categorical_fields: list,
        cat_dims: list,
        num_categorical_fields: int,
        num_numerical_fields: int,
        output_dim: int,
        backbone: str,
        loss_fn: Callable = F.cross_entropy,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        metrics: METRICS_TYPE = None,
        learning_rate: float = 1e-2,
        output: OUTPUT_TYPE = None,
        **backbone_kwargs
    ):
        self.save_hyperparameters()
        metadata = self.backbones.get(backbone, with_metadata=True)
        adapter = metadata["metadata"]["adapter"].from_task(
            self,
            task_type="regression",
            embedding_sizes=embedding_sizes,
            categorical_fields=categorical_fields,
            cat_dims=cat_dims,
            num_categorical_fields=num_categorical_fields,
            num_numerical_fields=num_numerical_fields,
            output_dim=output_dim,
            backbone=backbone,
            backbone_kwargs=backbone_kwargs,
            loss_fn=loss_fn,
            metrics=metrics,
        )
        super().__init__(
            adapter,
            loss_fn=loss_fn,
            metrics=metrics,
            learning_rate=learning_rate,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            output=output,
        )

    @classmethod
    def from_data(cls, datamodule, **kwargs) -> "TabularRegressor":
        model = cls(
            embedding_sizes=datamodule.embedding_sizes,
            categorical_fields=datamodule.categorical_fields,
            cat_dims=datamodule.cat_dims,
            num_categorical_fields=datamodule.num_categorical_fields,
            num_numerical_fields=datamodule.num_numerical_fields,
            output_dim=datamodule.output_dim,
            **kwargs
        )
        return model

    @requires("serve")
    def serve(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        sanity_check: bool = True,
        input_cls: Optional[Type[ServeInput]] = TabularDeserializer,
        transform: INPUT_TRANSFORM_TYPE = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Composition:
        return super().serve(
            host, port, sanity_check, partial(input_cls, parameters=parameters), transform, transform_kwargs
        )
