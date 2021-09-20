from copy import copy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Type, Union

import torch
import torchmetrics
from torch.optim.lr_scheduler import _LRScheduler

from flash import Task
from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.states import CollateFn
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _FORECASTING_AVAILABLE, _PANDAS_AVAILABLE
from flash.tabular.forecasting.backbones import TABULAR_FORECASTING_BACKBONES
from flash.tabular.forecasting.data import TabularForecastingData

if _PANDAS_AVAILABLE:
    from pandas.core.frame import DataFrame

if _FORECASTING_AVAILABLE:
    from pytorch_forecasting import TimeSeriesDataSet


class PatchTimeSeriesDataSet(TimeSeriesDataSet):
    """Hack to prevent index construction when instantiating model."""

    def _construct_index(self, data: DataFrame, predict_mode: bool) -> DataFrame:
        return DataFrame()


class TabularForecaster(Task):
    backbones: FlashRegistry = TABULAR_FORECASTING_BACKBONES

    def __init__(
        self,
        parameters: Dict[str, Any],
        backbone: str = "temporal_fusion_transformer",
        loss_fn: Optional[Callable] = None,
        optimizer: Union[Type[torch.optim.Optimizer], torch.optim.Optimizer, str] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Union[Type[_LRScheduler], str, _LRScheduler]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        metrics: Union[torchmetrics.Metric, List[torchmetrics.Metric]] = None,
        learning_rate: float = 3e-2,
        **backbone_kwargs
    ):
        super().__init__(
            model=None,
            loss_fn=None,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            metrics=None,
            learning_rate=learning_rate,
        )

        self.save_hyperparameters()

        parameters = copy(parameters)
        data = parameters.pop("data_sample")
        time_series_dataset = PatchTimeSeriesDataSet.from_parameters(parameters, data)

        backbone_kwargs["loss"] = loss_fn

        if metrics is not None and not isinstance(metrics, list):
            metrics = [metrics]
        backbone_kwargs["logging_metrics"] = metrics

        if not backbone_kwargs:
            backbone_kwargs = {}

        self.backbone = self.backbones.get(backbone)(time_series_dataset=time_series_dataset, **backbone_kwargs)

        self.set_state(CollateFn(partial(TabularForecaster._collate_fn, time_series_dataset._collate_fn)))

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return self.backbone.training_step(batch, batch_idx)

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return self.backbone.validation_step(batch, batch_idx)

    @classmethod
    def from_data(cls, tabular_forecasting_data: TabularForecastingData, **kwargs):
        return cls(tabular_forecasting_data=tabular_forecasting_data, **kwargs)

    @staticmethod
    def _collate_fn(collate_fn, samples):
        samples = [(sample[DefaultDataKeys.INPUT], sample[DefaultDataKeys.TARGET]) for sample in samples]
        batch = collate_fn(samples)
        return {DefaultDataKeys.INPUT: batch[0], DefaultDataKeys.TARGET: batch[1]}
