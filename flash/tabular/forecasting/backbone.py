from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _FORECASTING_AVAILABLE
from flash.tabular.forecasting.data import TabularForecastingData

if _FORECASTING_AVAILABLE:
    from pytorch_forecasting import TemporalFusionTransformer


TABULAR_FORECASTING_BACKBONES = FlashRegistry("backbones")


if _FORECASTING_AVAILABLE:

    @TABULAR_FORECASTING_BACKBONES(name="temporal_fusion_transformer", namespace="tabular/forecasting")
    def temporal_fusion_transformer(tabular_forecasting_data: TabularForecastingData, **kwargs):
        return TemporalFusionTransformer.from_dataset(
            tabular_forecasting_data.train_dataset.time_series_dataset, **kwargs
        )
