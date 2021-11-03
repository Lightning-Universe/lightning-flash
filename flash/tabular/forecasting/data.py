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
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from pytorch_lightning.utilities.exceptions import MisconfigurationException

from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_module import DataModule
from flash.core.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources
from flash.core.data.process import Deserializer, Preprocess
from flash.core.data.properties import ProcessState
from flash.core.utilities.imports import _FORECASTING_AVAILABLE, _PANDAS_AVAILABLE, requires

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

    time_series_dataset_parameters: Optional[Dict[str, Any]]


class TabularForecastingDataFrameDataSource(DataSource[DataFrame]):
    @requires("tabular")
    def __init__(
        self,
        time_idx: Optional[str] = None,
        target: Optional[Union[str, List[str]]] = None,
        group_ids: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        **data_source_kwargs: Any,
    ):
        super().__init__()
        self.time_idx = time_idx
        self.target = target
        self.group_ids = group_ids
        self.data_source_kwargs = data_source_kwargs

        self.set_state(TimeSeriesDataSetParametersState(parameters))

    def load_data(self, data: DataFrame, dataset: Optional[Any] = None):
        if self.training:
            time_series_dataset = TimeSeriesDataSet(
                data, time_idx=self.time_idx, group_ids=self.group_ids, target=self.target, **self.data_source_kwargs
            )
            parameters = time_series_dataset.get_parameters()

            # Add some sample data so that we can recreate the `TimeSeriesDataSet` later on
            parameters["data_sample"] = data.iloc[[0]]

            self.set_state(TimeSeriesDataSetParametersState(parameters))
            dataset.parameters = parameters
        else:
            parameters = copy(self.get_state(TimeSeriesDataSetParametersState).time_series_dataset_parameters)
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
        dataset.time_series_dataset = time_series_dataset
        return time_series_dataset

    def load_sample(self, sample: Mapping[str, Any], dataset: Optional[Any] = None) -> Any:
        return {DefaultDataKeys.INPUT: sample[0], DefaultDataKeys.TARGET: sample[1]}


class TabularForecastingPreprocess(Preprocess):
    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        deserializer: Optional[Deserializer] = None,
        **data_source_kwargs: Any,
    ):
        self.data_source_kwargs = data_source_kwargs
        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources={
                DefaultDataSources.DATAFRAME: TabularForecastingDataFrameDataSource(**data_source_kwargs),
            },
            deserializer=deserializer,
            default_data_source=DefaultDataSources.DATAFRAME,
        )

    def get_state_dict(self, strict: bool = False) -> Dict[str, Any]:
        return {**self.transforms, **self.data_source_kwargs}

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = True) -> "Preprocess":
        return cls(**state_dict)


class TabularForecastingData(DataModule):
    """Data module for the tabular forecasting task."""

    preprocess_cls = TabularForecastingPreprocess

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
        data_fetcher: Optional[BaseDataFetcher] = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        **preprocess_kwargs: Any,
    ):
        """Creates a :class:`~flash.tabular.forecasting.data.TabularForecastingData` object from the given data
        frames.

        .. note::

            The ``time_idx``, ``target``, and ``group_ids`` do not need to be provided if ``parameters`` are passed
            instead. These can be obtained from the
            :attr:`~flash.tabular.forecasting.data.TabularForecastingData.parameters` attribute of the
            :class:`~flash.tabular.forecasting.data.TabularForecastingData` object that contains your training data.

        Args:
            time_idx:
            target: Column denoting the target or list of columns denoting the target.
            group_ids: List of column names identifying a time series. This means that the group_ids identify a sample
                together with the time_idx. If you have only one timeseries, set this to the name of column that is
                constant.
            parameters: Parameters to use for the timeseries if ``time_idx``, ``target``, and ``group_ids`` are not
                provided (e.g. when loading data for inference or validation).
            train_data_frame: The pandas ``DataFrame`` containing the training data.
            val_data_frame: The pandas ``DataFrame`` containing the validation data.
            test_data_frame: The pandas ``DataFrame`` containing the testing data.
            predict_data_frame: The pandas ``DataFrame`` containing the data to use when predicting.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            data_fetcher: The :class:`~flash.core.data.callback.BaseDataFetcher` to pass to the
                :class:`~flash.core.data.data_module.DataModule`.
            preprocess: The :class:`~flash.core.data.data.Preprocess` to pass to the
                :class:`~flash.core.data.data_module.DataModule`. If ``None``, ``cls.preprocess_cls``
                will be constructed and used.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            batch_size: The ``batch_size`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            num_workers: The ``num_workers`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            preprocess_kwargs: Additional keyword arguments to use when constructing the preprocess. Will only be used
                if ``preprocess = None``.

        Returns:
            The constructed data module.

        Examples::

            data_module = TabularForecastingData.from_data_frame(
                time_idx="time_idx",
                target="value",
                group_ids=["series"],
                train_data_frame=train_data,
            )
        """

        return cls.from_data_source(
            time_idx=time_idx,
            target=target,
            group_ids=group_ids,
            parameters=parameters,
            data_source=DefaultDataSources.DATAFRAME,
            train_data=train_data_frame,
            val_data=val_data_frame,
            test_data=test_data_frame,
            predict_data=predict_data_frame,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            **preprocess_kwargs,
        )
