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
from flash.core.utilities.flash_cli import FlashCLI
from flash.core.utilities.imports import _FORECASTING_AVAILABLE, _PANDAS_AVAILABLE
from flash.tabular.forecasting.data import TabularForecastingData
from flash.tabular.forecasting.model import TabularForecaster

if _FORECASTING_AVAILABLE:
    from pytorch_forecasting.data import NaNLabelEncoder
    from pytorch_forecasting.data.examples import generate_ar_data

if _PANDAS_AVAILABLE:
    import pandas as pd

__all__ = ["tabular_forecasting"]


def from_synthetic_ar_data(
    seasonality: float = 10.0,
    timesteps: int = 400,
    n_series: int = 100,
    max_encoder_length: int = 60,
    max_prediction_length: int = 20,
    batch_size: int = 4,
    num_workers: int = 0,
    **time_series_dataset_kwargs,
) -> TabularForecastingData:
    """Creates and loads a synthetic Auto-Regressive (AR) data set."""
    data = generate_ar_data(seasonality=seasonality, timesteps=timesteps, n_series=n_series, seed=42)
    data["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(data.time_idx, "D")

    training_cutoff = data["time_idx"].max() - max_prediction_length

    return TabularForecastingData.from_data_frame(
        time_idx="time_idx",
        target="value",
        categorical_encoders={"series": NaNLabelEncoder().fit(data.series)},
        group_ids=["series"],
        # only unknown variable is "value" - and N-Beats can also not take any additional variables
        time_varying_unknown_reals=["value"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        train_data_frame=data[lambda x: x.time_idx <= training_cutoff],
        val_data_frame=data,
        batch_size=batch_size,
        num_workers=num_workers,
        **time_series_dataset_kwargs,
    )


def tabular_forecasting():
    """Timeseries forecasting."""
    cli = FlashCLI(
        TabularForecaster,
        TabularForecastingData,
        default_datamodule_builder=from_synthetic_ar_data,
        default_arguments={
            "trainer.max_epochs": 1,
            "model.backbone": "n_beats",
            "model.backbone_kwargs": {"widths": [32, 512], "backcast_loss_ratio": 0.1},
        },
        finetune=False,
        datamodule_attributes={"parameters"},
    )

    cli.trainer.save_checkpoint("tabular_forecasting_model.pt")


if __name__ == "__main__":
    tabular_forecasting()
