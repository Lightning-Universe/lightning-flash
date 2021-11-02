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
import functools

from flash.core.integrations.pytorch_forecasting.adapter import PyTorchForecastingAdapter
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _FORECASTING_AVAILABLE
from flash.core.utilities.providers import _PYTORCH_FORECASTING

if _FORECASTING_AVAILABLE:
    from pytorch_forecasting import (
        DecoderMLP,
        DeepAR,
        NBeats,
        RecurrentNetwork,
        TemporalFusionTransformer,
        TimeSeriesDataSet,
    )


PYTORCH_FORECASTING_BACKBONES = FlashRegistry("backbones")


if _FORECASTING_AVAILABLE:

    def load_torch_forecasting(model, time_series_dataset: TimeSeriesDataSet, **kwargs):
        return model.from_dataset(time_series_dataset, **kwargs)

    for model, name in zip(
        [TemporalFusionTransformer, NBeats, RecurrentNetwork, DeepAR, DecoderMLP],
        ["temporal_fusion_transformer", "n_beats", "recurrent_network", "deep_ar", "decoder_mlp"],
    ):
        PYTORCH_FORECASTING_BACKBONES(
            functools.partial(load_torch_forecasting, model),
            name=name,
            providers=_PYTORCH_FORECASTING,
            adapter=PyTorchForecastingAdapter,
        )
