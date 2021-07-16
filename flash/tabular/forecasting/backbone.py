from pytorch_forecasting import TemporalFusionTransformer

from flash.core.registry import FlashRegistry
from flash.tabular.forecasting import TabularForecastingData

TABULAR_FORECASTING_BACKBONES = FlashRegistry("backbones")


@TABULAR_FORECASTING_BACKBONES(name="temporal_fusion_transformer", namespace="tabular/forecasting")
def temporal_fusion_transformer(tabular_forecasting_data: TabularForecastingData, **kwargs):
    return TemporalFusionTransformer.from_dataset(
        tabular_forecasting_data.train_dataset.time_series_dataset,
        **kwargs
    )
