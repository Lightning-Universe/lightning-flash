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
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data.sampler import Sampler

from flash.core.data.data_module import DataModule
from flash.core.data.data_pipeline import DataPipelineState
from flash.core.data.io.input import DataKeys, Input, InputFormat
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.process import Deserializer
from flash.core.data.properties import ProcessState
from flash.core.utilities.imports import _FORECASTING_AVAILABLE, _PANDAS_AVAILABLE, requires
from flash.core.utilities.stages import RunningStage

if _PANDAS_AVAILABLE:
    from pandas.core.frame import DataFrame
else:
    DataFrame = object

if _FORECASTING_AVAILABLE:
    from pytorch_forecasting import TimeSeriesDataSet


@dataclass(unsafe_hash=True, frozen=True)
class TimeSeriesDataSetParametersState(ProcessState):
    """A :class:`~flash.core.data.properties.ProcessState` containing ``labels``, a mapping from class index to
    label."""

    parameters: Optional[Dict[str, Any]]


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

            self.set_state(TimeSeriesDataSetParametersState(parameters))
            self.parameters = parameters
        else:
            parameters_state = self.get_state(TimeSeriesDataSetParametersState)
            parameters = parameters or (parameters_state.parameters if parameters_state is not None else None)
            parameters = copy(parameters)
            if parameters is None:
                raise MisconfigurationException(
                    "Loading data for evaluation or inference requires parameters from the train data. Either "
                    "construct the train data at the same time as evaluation and inference or provide the train "
                    "`datamodule.parameters` to `from_data_frame` in the `parameters` argument."
                )
            parameters.pop("data_sample")
            time_series_dataset = TimeSeriesDataSet.from_parameters(
                parameters,
                data,
                predict=True,
                stop_randomization=True,
            )
        return time_series_dataset

    def load_sample(self, sample: Tuple) -> Any:
        return {DataKeys.INPUT: sample[0], DataKeys.TARGET: sample[1]}


class TabularForecastingInputTransform(InputTransform):
    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        deserializer: Optional[Deserializer] = None,
    ):
        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            inputs={
                InputFormat.DATAFRAME: TabularForecastingDataFrameInput,
            },
            deserializer=deserializer,
            default_input=InputFormat.DATAFRAME,
        )

    def get_state_dict(self, strict: bool = False) -> Dict[str, Any]:
        return {**self.transforms}

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = True) -> "InputTransform":
        return cls(**state_dict)


class TabularForecastingData(DataModule):
    """Data module for the tabular forecasting task."""

    input_transform_cls = TabularForecastingInputTransform

    @property
    def parameters(self) -> Optional[Dict[str, Any]]:
        """The parameters dictionary from the ``TimeSeriesDataSet`` object created from the train data when
        constructing the ``TabularForecastingData`` object."""
        return getattr(self.train_dataset, "parameters", None)

    @classmethod
    def from_data_frame(
        cls,
        time_idx: Optional[str] = None,
        target: Optional[Union[str, List[str]]] = None,
        group_ids: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        train_data_frame: Optional[DataFrame] = None,
        val_data_frame: Optional[DataFrame] = None,
        test_data_frame: Optional[DataFrame] = None,
        predict_data_frame: Optional[DataFrame] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        # TODO: Update these when DataModule is updated
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: int = 0,
        sampler: Optional[Type[Sampler]] = None,
        **time_series_dataset_kwargs: Any,
    ) -> "TabularForecastingData":
        """Creates a :class:`~flash.tabular.forecasting.data.TabularForecastingData` object from the given data
        frames.

        .. note::

            The ``time_idx``, ``target``, and ``group_ids`` do not need to be provided if ``parameters`` are passed
            instead. These can be obtained from the
            :attr:`~flash.tabular.forecasting.data.TabularForecastingData.parameters` attribute of the
            :class:`~flash.tabular.forecasting.data.TabularForecastingData` object that contains your training data.
        """

        data_pipeline_state = DataPipelineState()

        train_input = TabularForecastingDataFrameInput(
            RunningStage.TRAINING,
            train_data_frame,
            time_idx=time_idx,
            group_ids=group_ids,
            target=target,
            **time_series_dataset_kwargs,
            data_pipeline_state=data_pipeline_state,
        )

        dataset_kwargs = dict(
            data_pipeline_state=data_pipeline_state, parameters=train_input.parameters if train_input else parameters
        )
        return cls(
            train_input,
            TabularForecastingDataFrameInput(RunningStage.VALIDATING, val_data_frame, **dataset_kwargs),
            TabularForecastingDataFrameInput(RunningStage.TESTING, test_data_frame, **dataset_kwargs),
            TabularForecastingDataFrameInput(RunningStage.PREDICTING, predict_data_frame, **dataset_kwargs),
            input_transform=cls.input_transform_cls(train_transform, val_transform, test_transform, predict_transform),
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
        )
