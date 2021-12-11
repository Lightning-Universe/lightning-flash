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

from torch.utils.data.sampler import Sampler

from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_module import DataModule
from flash.core.data.data_pipeline import DataPipelineState
from flash.core.data.input_transform import INPUT_TRANSFORM_TYPE, InputTransform
from flash.core.data.io.input import Input
from flash.core.data.io.output_transform import OutputTransform
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
        transform_kwargs: Optional[Dict] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        val_split: Optional[float] = None,
        batch_size: Optional[int] = None,
        num_workers: int = 0,
        sampler: Optional[Type[Sampler]] = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        output_transform: Optional[OutputTransform] = None,
        **input_kwargs: Any,
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
            **input_kwargs,
        )

        train_input = input_cls(RunningStage.TRAINING, train_data_frame, transform=train_transform, **ds_kw)

        ds_kw["parameters"] = train_input.parameters if train_input else parameters

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_data_frame, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_data_frame, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_data_frame, transform=predict_transform, **ds_kw),
            data_fetcher=data_fetcher,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            output_transform=output_transform,
        )
