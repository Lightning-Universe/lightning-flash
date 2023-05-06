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
from copy import copy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

import torchmetrics
from torch import Tensor

from flash.core.adapter import Adapter
from flash.core.data.batch import default_uncollate
from flash.core.data.io.input import DataKeys
from flash.core.model import Task
from flash.core.utilities.imports import _FORECASTING_AVAILABLE, _PANDAS_AVAILABLE

if _PANDAS_AVAILABLE:
    from pandas import DataFrame
else:
    DataFrame = object

if _FORECASTING_AVAILABLE:
    from pytorch_forecasting import TimeSeriesDataSet
else:
    TimeSeriesDataSet = object


class PatchTimeSeriesDataSet(TimeSeriesDataSet):
    """Hack to prevent index construction or data validation / conversion when instantiating model.

    This enables the ``TimeSeriesDataSet`` to be created from a single row of data.
    """

    def _construct_index(self, data: DataFrame, predict_mode: bool) -> DataFrame:
        return DataFrame()

    def _data_to_tensors(self, data: DataFrame) -> Dict[str, Tensor]:
        return {}


class PyTorchForecastingAdapter(Adapter):
    """The ``PyTorchForecastingAdapter`` is an :class:`~flash.core.adapter.Adapter` for integrating with PyTorch
    Forecasting."""

    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone

    @staticmethod
    def _collate_fn(collate_fn, samples):
        samples = [(sample[DataKeys.INPUT], sample[DataKeys.TARGET]) for sample in samples]
        batch = collate_fn(samples)
        return {DataKeys.INPUT: batch[0], DataKeys.TARGET: batch[1]}

    @classmethod
    def from_task(
        cls,
        task: Task,
        parameters: Dict[str, Any],
        backbone: str,
        backbone_kwargs: Optional[Dict[str, Any]] = None,
        loss_fn: Optional[Callable] = None,
        metrics: Optional[Union[torchmetrics.Metric, List[torchmetrics.Metric]]] = None,
    ) -> Adapter:
        parameters = copy(parameters)
        # Remove the single row of data from the parameters to reconstruct the `time_series_dataset`
        data = DataFrame.from_dict(parameters.pop("data_sample"))
        time_series_dataset = PatchTimeSeriesDataSet.from_parameters(parameters, data)

        backbone_kwargs["loss"] = loss_fn

        if metrics is not None and not isinstance(metrics, list):
            metrics = [metrics]
        backbone_kwargs["logging_metrics"] = metrics

        backbone_kwargs = backbone_kwargs or {}

        adapter = cls(task.backbones.get(backbone)(time_series_dataset=time_series_dataset, **backbone_kwargs))

        # Attach the required collate function
        adapter.collate_fn = partial(PyTorchForecastingAdapter._collate_fn, time_series_dataset._collate_fn)

        return adapter

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DataKeys.INPUT], batch[DataKeys.TARGET])
        return self.backbone.training_step(batch, batch_idx)

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DataKeys.INPUT], batch[DataKeys.TARGET])
        return self.backbone.validation_step(batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        raise NotImplementedError(
            "Backbones provided by PyTorch Forecasting don't support testing. Use validation instead."
        )

    def forward(self, x: Any) -> Any:
        return dict(self.backbone(x))

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        result = self(batch[DataKeys.INPUT])
        result[DataKeys.INPUT] = default_uncollate(batch[DataKeys.INPUT])
        return default_uncollate(result)

    def training_epoch_end(self, outputs) -> None:
        self.backbone.training_epoch_end(outputs)

    def validation_epoch_end(self, outputs) -> None:
        self.backbone.validation_epoch_end(outputs)
