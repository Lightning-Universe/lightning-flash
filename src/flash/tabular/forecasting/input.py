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
from typing import Any, Dict, List, Optional, Tuple, Union

from flash.core.data.io.input import DataKeys, Input
from flash.core.utilities.imports import _FORECASTING_AVAILABLE, _PANDAS_AVAILABLE, requires

if _PANDAS_AVAILABLE:
    from pandas.core.frame import DataFrame
else:
    DataFrame = object

if _FORECASTING_AVAILABLE:
    from pytorch_forecasting import TimeSeriesDataSet


class TabularForecastingDataFrameInput(Input):
    @requires("tabular")
    def load_data(
        self,
        data: DataFrame,
        time_idx: Optional[str] = None,
        target: Optional[Union[str, List[str]]] = None,
        group_ids: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        **time_series_dataset_kwargs: Any,
    ):
        if self.training:
            time_series_dataset = TimeSeriesDataSet(
                data, time_idx=time_idx, group_ids=group_ids, target=target, **time_series_dataset_kwargs
            )
            parameters = time_series_dataset.get_parameters()

            # Add some sample data so that we can recreate the `TimeSeriesDataSet` later on
            parameters["data_sample"] = data.iloc[[0]].to_dict()

            self.parameters = parameters
        else:
            if parameters is None:
                raise ValueError(
                    "Loading data for evaluation or inference requires parameters from the train data. Either "
                    "construct the train data at the same time as evaluation and inference or provide the train "
                    "`datamodule.parameters` to `from_data_frame` in the `parameters` argument."
                )
            parameters = copy(parameters)
            parameters.pop("data_sample")
            time_series_dataset = TimeSeriesDataSet.from_parameters(
                parameters,
                data,
                stop_randomization=True,
            )
        return time_series_dataset

    def load_sample(self, sample: Tuple) -> Any:
        return {DataKeys.INPUT: sample[0], DataKeys.TARGET: sample[1]}
