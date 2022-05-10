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
from typing import Any, Callable, Dict, Optional, Type, Union

from torch.nn import functional as F

from flash.core.data.io.input import ServeInput
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.io.output import Output
from flash.core.integrations.pytorch_tabular.backbones import PYTORCH_TABULAR_BACKBONES
from flash.core.registry import FlashRegistry
from flash.core.regression import RegressionAdapterTask
from flash.core.serve import Composition
from flash.core.utilities.imports import _TABULAR_TESTING, requires
from flash.core.utilities.types import INPUT_TRANSFORM_TYPE, LR_SCHEDULER_TYPE, METRICS_TYPE, OPTIMIZER_TYPE
from flash.tabular.input import TabularDeserializer

# Skip doctests if requirements aren't available
if not _TABULAR_TESTING:
    __doctest_skip__ = ["TabularRegressor", "TabularRegressor.*"]


class TabularRegressor(RegressionAdapterTask):
    """The ``TabularRegressor`` is a :class:`~flash.Task` for classifying tabular data. For more details, see
    :ref:`tabular_classification`.

    Args:
        parameters: The parameters computed from the training data (can be obtained from the ``parameters`` attribute of
            the ``TabularRegressionData`` object containing your training data).
        embedding_sizes: List of (num_classes, emb_dim) to form categorical embeddings.
        cat_dims: Number of distinct values for each categorical column
        num_features: Number of columns in table
        backbone: name of the model to use
        loss_fn: Loss function for training, defaults to cross entropy.
        optimizer: Optimizer to use for training.
        lr_scheduler: The LR scheduler to use during training.
        metrics: Metrics to compute for training and evaluation. Can either be an metric from the `torchmetrics`
            package, a custom metric inherenting from `torchmetrics.Metric`, a callable function or a list/dict
            containing a combination of the aforementioned. In all cases, each metric needs to have the signature
            `metric(preds,target)` and return a single scalar tensor. Defaults to :class:`torchmetrics.Accuracy`.
        learning_rate: Learning rate to use for training.
        **backbone_kwargs: Optional additional arguments for the model.
    """

    required_extras: str = "tabular"
    backbones: FlashRegistry = FlashRegistry("backbones") + PYTORCH_TABULAR_BACKBONES

    def __init__(
        self,
        parameters: Dict[str, Any],
        embedding_sizes: list,
        cat_dims: list,
        num_features: int,
        backbone: str = "tabnet",
        loss_fn: Callable = F.mse_loss,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        metrics: METRICS_TYPE = None,
        learning_rate: Optional[float] = None,
        **backbone_kwargs
    ):
        self.save_hyperparameters()

        self._data_parameters = parameters

        metadata = self.backbones.get(backbone, with_metadata=True)
        adapter = metadata["metadata"]["adapter"].from_task(
            self,
            task_type="regression",
            embedding_sizes=embedding_sizes,
            categorical_fields=parameters["categorical_fields"],
            cat_dims=cat_dims,
            num_features=num_features,
            output_dim=1,
            backbone=backbone,
            backbone_kwargs=backbone_kwargs,
            loss_fn=loss_fn,
            metrics=metrics,
        )
        super().__init__(
            adapter,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            learning_rate=learning_rate,
        )

    @property
    def data_parameters(self) -> Dict[str, Any]:
        """Get the parameters computed from the training data used to create this
        :class:`~flash.tabular.regression.TabularRegressor`. Use these parameters to load data for
        evaluation / prediction.

        Examples
        ________

        .. doctest::

            >>> import flash
            >>> from flash.core.data.utils import download_data
            >>> from flash.tabular import TabularRegressionData, TabularRegressor
            >>> download_data("https://pl-flash-data.s3.amazonaws.com/SeoulBikeData.csv", "./data")
            >>> model = TabularRegressor.load_from_checkpoint(
            ...     "https://flash-weights.s3.amazonaws.com/0.7.0/tabular_regression_model.pt"
            ... )
            >>> datamodule = TabularRegressionData.from_csv(
            ...     predict_file="data/SeoulBikeData.csv",
            ...     parameters=model.data_parameters,
            ...     batch_size=8,
            ... )
            >>> trainer = flash.Trainer()
            >>> trainer.predict(
            ...     model,
            ...     datamodule=datamodule,
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...
        """
        return self._data_parameters

    @classmethod
    def from_data(cls, datamodule, **kwargs) -> "TabularRegressor":
        model = cls(
            parameters=datamodule.parameters,
            embedding_sizes=datamodule.embedding_sizes,
            cat_dims=datamodule.cat_dims,
            num_features=datamodule.num_features,
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
        output: Optional[Union[str, Output]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Composition:
        parameters = parameters or self._data_parameters
        return super().serve(
            host, port, sanity_check, partial(input_cls, parameters=parameters), transform, transform_kwargs, output
        )
