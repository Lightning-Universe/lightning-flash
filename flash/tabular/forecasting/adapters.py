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

from flash.core.adapter import Adapter
from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.states import CollateFn
from flash.core.model import Task
from flash.core.utilities.imports import _FORECASTING_AVAILABLE, _PANDAS_AVAILABLE

if _PANDAS_AVAILABLE:
    from pandas.core.frame import DataFrame

if _FORECASTING_AVAILABLE:
    from pytorch_forecasting import TimeSeriesDataSet
else:
    TimeSeriesDataSet = object


class PatchTimeSeriesDataSet(TimeSeriesDataSet):
    """Hack to prevent index construction when instantiating model.

    This enables the ``TimeSeriesDataSet`` to be created from a single row of data.
    """

    def _construct_index(self, data: DataFrame, predict_mode: bool) -> DataFrame:
        return DataFrame()


class PyTorchForecastingAdapter(Adapter):
    """The ``PyTorchForecastingAdapter`` is an :class:`~flash.core.adapter.Adapter` for integrating with PyTorch
    Forecasting."""

    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone

    @staticmethod
    def _collate_fn(collate_fn, samples):
        samples = [(sample[DefaultDataKeys.INPUT], sample[DefaultDataKeys.TARGET]) for sample in samples]
        batch = collate_fn(samples)
        return {DefaultDataKeys.INPUT: batch[0], DefaultDataKeys.TARGET: batch[1]}

    @classmethod
    def from_task(
        cls,
        task: Task,
        parameters: Dict[str, Any],
        backbone: str,
        loss_fn: Optional[Callable] = None,
        metrics: Optional[Union[torchmetrics.Metric, List[torchmetrics.Metric]]] = None,
        **backbone_kwargs,
    ) -> Adapter:
        parameters = copy(parameters)
        data = parameters.pop("data_sample")
        time_series_dataset = PatchTimeSeriesDataSet.from_parameters(parameters, data)

        backbone_kwargs["loss"] = loss_fn

        if metrics is not None and not isinstance(metrics, list):
            metrics = [metrics]
        backbone_kwargs["logging_metrics"] = metrics

        if not backbone_kwargs:
            backbone_kwargs = {}

        forecasting_model = task.backbones.get(backbone)(time_series_dataset=time_series_dataset, **backbone_kwargs)

        # Attach the required collate function
        task.set_state(CollateFn(partial(PyTorchForecastingAdapter._collate_fn, time_series_dataset._collate_fn)))

        # Attach the `forecasting_model` attribute to expose the built-in inference methods from PyTorch Forecasting
        task.forecasting_model = forecasting_model

        return cls(forecasting_model)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return self.backbone.training_step(batch, batch_idx)

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return self.backbone.validation_step(batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        # PyTorch Forecasting models don't have a `test_step`, so re-use `validation_step`
        return self.backbone.validation_step(batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        raise NotImplementedError(
            "Flash's inference is not currently supported with backbones provided by PyTorch Forecasting. You can "
            "access the PyTorch Forecasting LightningModule directly with the `forecasting_model` attribute of the "
            "`TabularForecaster`."
        )

    def training_epoch_end(self, outputs) -> None:
        self.backbone.training_epoch_end(outputs)

    def validation_epoch_end(self, outputs) -> None:
        self.backbone.validation_epoch_end(outputs)

    def test_epoch_end(self, outputs) -> None:
        # PyTorch Forecasting models don't have a `test_epoch_end`, so re-use `validation_epoch_end`
        self.backbone.validation_epoch_end(outputs)
