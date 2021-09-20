import functools

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _FORECASTING_AVAILABLE

if _FORECASTING_AVAILABLE:
    from pytorch_forecasting import (
        DecoderMLP,
        DeepAR,
        NBeats,
        RecurrentNetwork,
        TemporalFusionTransformer,
        TimeSeriesDataSet,
    )


TABULAR_FORECASTING_BACKBONES = FlashRegistry("backbones")


if _FORECASTING_AVAILABLE:

    def load_torch_forecasting(model, time_series_dataset: TimeSeriesDataSet, **kwargs):
        return model.from_dataset(time_series_dataset, **kwargs)

    for model, name in zip(
        [TemporalFusionTransformer, NBeats, RecurrentNetwork, DeepAR, DecoderMLP],
        ["temporal_fusion_transformer", "n_beats", "recurrent_network", "deep_ar", "decoder_mlp"],
    ):
        TABULAR_FORECASTING_BACKBONES(
            functools.partial(load_torch_forecasting, model),
            name=name,
            namespace="tabular/forecasting",
        )
