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
from typing import Any, Callable, Dict, List, Optional, Type, Union

from flash.core.data.data_pipeline import DataPipelineState
from flash.core.data.input_transform import INPUT_TRANSFORM_TYPE, InputTransform
from flash.core.data.io.input import InputFormat
from flash.core.data.io.input_base import Input
from flash.core.data.io.output_transform import OutputTransform
from flash.core.data.new_data_module import DataModule
from flash.core.data.process import Deserializer
from flash.core.utilities.imports import _PANDAS_AVAILABLE
from flash.core.utilities.stages import RunningStage
from flash.tabular.input import TabularCSVInput, TabularDataFrameInput, TabularDeserializer

if _PANDAS_AVAILABLE:
    from pandas.core.frame import DataFrame
else:
    DataFrame = object


class TabularInputTransform(InputTransform):
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
                InputFormat.CSV: TabularCSVInput,
                InputFormat.DATAFRAME: TabularDataFrameInput,
            },
            default_input=InputFormat.CSV,
            deserializer=deserializer or TabularDeserializer(),
        )

    def get_state_dict(self, strict: bool = False) -> Dict[str, Any]:
        return self.transforms

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = True) -> "InputTransform":
        return cls(**state_dict)


class TabularData(DataModule):
    """Data module for tabular tasks."""

    input_transform_cls = TabularInputTransform
    output_transform_cls = OutputTransform

    is_regression: bool = False

    @property
    def parameters(self) -> Optional[Dict[str, Any]]:
        """The parameters dictionary created from the train data when constructing the ``TabularData`` object."""
        return getattr(self.train_dataset, "parameters", None)

    @property
    def codes(self) -> Dict[str, str]:
        return self.parameters["codes"]

    @property
    def categorical_fields(self) -> Optional[List[str]]:
        return self.parameters["categorical_fields"]

    @property
    def numerical_fields(self) -> Optional[List[str]]:
        return self.parameters["numerical_fields"]

    @property
    def num_features(self) -> int:
        return len(self.categorical_fields) + len(self.numerical_fields)

    @property
    def embedding_sizes(self) -> list:
        """Recommended embedding sizes."""

        # https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
        # The following "formula" provides a general rule of thumb about the number of embedding dimensions:
        # embedding_dimensions =  number_of_categories**0.25
        num_classes = [len(self.codes[cat]) + 1 for cat in self.categorical_fields]
        emb_dims = [max(int(n ** 0.25), 16) for n in num_classes]
        return list(zip(num_classes, emb_dims))

    @classmethod
    def from_data_frame(
        cls,
        categorical_fields: Optional[Union[str, List[str]]] = None,
        numerical_fields: Optional[Union[str, List[str]]] = None,
        target_fields: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        train_data_frame: Optional[DataFrame] = None,
        val_data_frame: Optional[DataFrame] = None,
        test_data_frame: Optional[DataFrame] = None,
        predict_data_frame: Optional[DataFrame] = None,
        train_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        input_cls: Type[Input] = TabularDataFrameInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "TabularData":

        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
            categorical_fields=categorical_fields,
            numerical_fields=numerical_fields,
            target_field=target_fields,
            is_regression=cls.is_regression,
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

    @classmethod
    def from_csv(
        cls,
        categorical_fields: Optional[Union[str, List[str]]] = None,
        numerical_fields: Optional[Union[str, List[str]]] = None,
        target_fields: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        train_file: Optional[str] = None,
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        predict_file: Optional[str] = None,
        train_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        input_cls: Type[Input] = TabularCSVInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "TabularData":

        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
            categorical_fields=categorical_fields,
            numerical_fields=numerical_fields,
            target_field=target_fields,
            is_regression=cls.is_regression,
        )

        train_input = input_cls(RunningStage.TRAINING, train_file, transform=train_transform, **ds_kw)

        ds_kw["parameters"] = train_input.parameters if train_input else parameters

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_file, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_file, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_file, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )
