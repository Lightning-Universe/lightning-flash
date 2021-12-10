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
from typing import Any, Dict, List, Optional, Type, Union

from flash.core.data.data_pipeline import DataPipelineState
from flash.core.data.input_transform import INPUT_TRANSFORM_TYPE, InputTransform
from flash.core.data.io.input_base import Input
from flash.core.data.new_data_module import DataModule
from flash.core.utilities.imports import _PANDAS_AVAILABLE
from flash.core.utilities.stages import RunningStage
from flash.tabular.forecasting.input import TabularForecastingDataFrameInput

if _PANDAS_AVAILABLE:
    from pandas.core.frame import DataFrame
else:
    DataFrame = object


class TabularForecastingData(DataModule):
    """Data module for the tabular forecasting task."""

    input_transform_cls = InputTransform

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
        train_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        input_cls: Type[Input] = TabularForecastingDataFrameInput,
        input_kwargs: Optional[Dict] = None,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "TabularForecastingData":
        """Creates a :class:`~flash.tabular.forecasting.data.TabularForecastingData` object from the given data
        frames.

        .. note::

            The ``time_idx``, ``target``, and ``group_ids`` do not need to be provided if ``parameters`` are passed
            instead. These can be obtained from the
            :attr:`~flash.tabular.forecasting.data.TabularForecastingData.parameters` attribute of the
            :class:`~flash.tabular.forecasting.data.TabularForecastingData` object that contains your training data.
        """

        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
            time_idx=time_idx,
            group_ids=group_ids,
            target=target,
            parameters=parameters,
            **(input_kwargs or {}),
        )

        train_input = input_cls(RunningStage.TRAINING, train_data_frame, transform=train_transform, **ds_kw)

        ds_kw["parameters"] = train_input.parameters if train_input else parameters

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_data_frame, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_data_frame, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_data_frame, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )
