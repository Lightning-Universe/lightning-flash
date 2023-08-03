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
from unittest.mock import MagicMock, patch

import pytest
from flash.core.utilities.imports import _TOPIC_TABULAR_AVAILABLE
from flash.tabular.forecasting import TabularForecastingData


@pytest.mark.skipif(not _TOPIC_TABULAR_AVAILABLE, reason="Tabular libraries aren't installed.")
@patch("flash.tabular.forecasting.input.TimeSeriesDataSet")
def test_from_data_frame_time_series_data_set_single_call(patch_time_series_data_set):
    """Tests that ``TabularForecastingData.from_data_frame`` calls ``TimeSeriesDataSet`` with the expected parameters
    when called once with data for all stages."""
    patch_time_series_data_set.return_value.get_parameters.return_value = {"test": None}

    train_data = MagicMock()
    val_data = MagicMock()

    TabularForecastingData.from_data_frame(
        "time_idx",
        "target",
        ["series"],
        train_data_frame=train_data,
        val_data_frame=val_data,
        additional_kwarg="test",
        batch_size=4,
    )

    patch_time_series_data_set.assert_called_once_with(
        train_data, time_idx="time_idx", group_ids=["series"], target="target", additional_kwarg="test"
    )

    patch_time_series_data_set.from_parameters.assert_called_once_with(
        {"test": None}, val_data, stop_randomization=True
    )


@pytest.mark.skipif(not _TOPIC_TABULAR_AVAILABLE, reason="Tabular libraries aren't installed.")
@patch("flash.tabular.forecasting.input.TimeSeriesDataSet")
def test_from_data_frame_time_series_data_set_multi_call(patch_time_series_data_set):
    """Tests that ``TabularForecastingData.from_data_frame`` calls ``TimeSeriesDataSet`` with the expected parameters
    when called separately for each stage."""
    patch_time_series_data_set.return_value.get_parameters.return_value = {"test": None}

    train_data = MagicMock()
    val_data = MagicMock()

    train_datamodule = TabularForecastingData.from_data_frame(
        "time_idx",
        "target",
        ["series"],
        train_data_frame=train_data,
        additional_kwarg="test",
        batch_size=4,
    )

    TabularForecastingData.from_data_frame(
        val_data_frame=val_data,
        parameters=train_datamodule.parameters,
        batch_size=4,
    )

    patch_time_series_data_set.assert_called_once_with(
        train_data, time_idx="time_idx", group_ids=["series"], target="target", additional_kwarg="test"
    )

    patch_time_series_data_set.from_parameters.assert_called_once_with(
        {"test": None}, val_data, stop_randomization=True
    )


@pytest.mark.skipif(not _TOPIC_TABULAR_AVAILABLE, reason="Tabular libraries aren't installed.")
def test_from_data_frame_misconfiguration():
    """Tests that a ``ValueError`` is raised when ``TabularForecastingData`` is constructed without parameters."""
    with pytest.raises(ValueError, match="evaluation or inference requires parameters"):
        TabularForecastingData.from_data_frame(
            "time_idx",
            "target",
            ["series"],
            val_data_frame=MagicMock(),
            additional_kwarg="test",
            batch_size=4,
        )
