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
import pytest

import flash
from flash.core.utilities.imports import _PANDAS_AVAILABLE, _TABULAR_AVAILABLE
from flash.tabular.forecasting import TabularForecaster, TabularForecastingData
from tests.helpers.utils import _TABULAR_TESTING

if _TABULAR_AVAILABLE:
    from pytorch_forecasting.data import NaNLabelEncoder
    from pytorch_forecasting.data.examples import generate_ar_data

if _PANDAS_AVAILABLE:
    import pandas as pd


@pytest.fixture
def sample_data():
    data = generate_ar_data(seasonality=10.0, timesteps=100, n_series=2, seed=42)
    data["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(data.time_idx, "D")
    max_prediction_length = 20
    training_cutoff = data["time_idx"].max() - max_prediction_length
    return data, training_cutoff, max_prediction_length


@pytest.mark.skipif(not _TABULAR_TESTING, reason="Tabular libraries aren't installed.")
def test_fast_dev_run_smoke(sample_data):
    """Test that fast dev run works with the NBeats example data."""
    data, training_cutoff, max_prediction_length = sample_data
    datamodule = TabularForecastingData.from_data_frame(
        time_idx="time_idx",
        target="value",
        categorical_encoders={"series": NaNLabelEncoder().fit(data.series)},
        group_ids=["series"],
        time_varying_unknown_reals=["value"],
        max_encoder_length=60,
        max_prediction_length=max_prediction_length,
        train_data_frame=data[lambda x: x.time_idx <= training_cutoff],
        val_data_frame=data,
    )

    model = TabularForecaster(datamodule.parameters, backbone="n_beats", widths=[32, 512], backcast_loss_ratio=0.1)

    trainer = flash.Trainer(max_epochs=1, fast_dev_run=True, gradient_clip_val=0.01)
    trainer.fit(model, datamodule=datamodule)


@pytest.mark.skipif(not _TABULAR_TESTING, reason="Tabular libraries aren't installed.")
def test_testing_raises(sample_data):
    """Tests that ``NotImplementedError`` is raised when attempting to perform a test pass."""
    data, training_cutoff, max_prediction_length = sample_data
    datamodule = TabularForecastingData.from_data_frame(
        time_idx="time_idx",
        target="value",
        categorical_encoders={"series": NaNLabelEncoder().fit(data.series)},
        group_ids=["series"],
        time_varying_unknown_reals=["value"],
        max_encoder_length=60,
        max_prediction_length=max_prediction_length,
        train_data_frame=data[lambda x: x.time_idx <= training_cutoff],
        test_data_frame=data,
    )

    model = TabularForecaster(datamodule.parameters, backbone="n_beats", widths=[32, 512], backcast_loss_ratio=0.1)
    trainer = flash.Trainer(max_epochs=1, fast_dev_run=True, gradient_clip_val=0.01)

    with pytest.raises(NotImplementedError, match="Backbones provided by PyTorch Forecasting don't support testing."):
        trainer.test(model, datamodule)
