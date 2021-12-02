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
from dataclasses import dataclass
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from flash.core.data.data_module import DataModule
from flash.core.data.data_pipeline import DataPipelineState
from flash.core.data.io.classification_input import ClassificationState
from flash.core.data.io.input import DataKeys, InputFormat
from flash.core.data.io.input_base import Input
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.io.output_transform import OutputTransform
from flash.core.data.process import Deserializer
from flash.core.data.properties import ProcessState
from flash.core.data.utilities.data_frame import read_csv
from flash.core.utilities.imports import _PANDAS_AVAILABLE
from flash.core.utilities.stages import RunningStage
from flash.tabular.classification.utils import (
    _compute_normalization,
    _generate_codes,
    _pre_transform,
    _to_cat_vars_numpy,
    _to_num_vars_numpy,
)

if _PANDAS_AVAILABLE:
    from pandas.core.frame import DataFrame
else:
    DataFrame = object


@dataclass(unsafe_hash=True, frozen=True)
class TabularParametersState(ProcessState):
    """A :class:`~flash.core.data.properties.ProcessState` containing tabular data ``parameters``."""

    parameters: Optional[Dict[str, Any]]


class TabularDataFrameInput(Input):
    @staticmethod
    def _sanetize_fields(
        categorical_fields: Optional[Union[str, List[str]]], numerical_fields: Optional[Union[str, List[str]]]
    ):
        if categorical_fields is None and numerical_fields is None:
            raise RuntimeError("Both `categorical_fields` and `numerical_fields` are None!")

        categorical_fields = categorical_fields or []
        numerical_fields = numerical_fields or []

        if not isinstance(categorical_fields, list):
            categorical_fields = [categorical_fields]

        if not isinstance(numerical_fields, list):
            numerical_fields = [numerical_fields]

        return categorical_fields, numerical_fields

    @staticmethod
    def compute_parameters(
        train_data_frame: DataFrame,
        target_field: str,
        numerical_fields: List[str],
        categorical_fields: List[str],
        is_regression: bool,
    ) -> Dict[str, Any]:

        mean, std = _compute_normalization(train_data_frame, numerical_fields)

        classes = list(train_data_frame[target_field].unique())

        if train_data_frame[target_field].dtype == object:
            # if the target_fields is a category, not an int
            target_codes = _generate_codes(train_data_frame, [target_field])
        else:
            target_codes = None
        codes = _generate_codes(train_data_frame, categorical_fields)

        return dict(
            mean=mean,
            std=std,
            classes=classes,
            codes=codes,
            target_codes=target_codes,
            target_field=target_field,
            numerical_fields=numerical_fields,
            categorical_fields=categorical_fields,
            is_regression=is_regression,
        )

    def load_data(
        self,
        df: DataFrame,
        categorical_fields: Optional[List[str]] = None,
        numerical_fields: Optional[List[str]] = None,
        target_field: Optional[str] = None,
        is_regression: bool = True,
        parameters: Dict[str, Any] = None,
    ):
        if self.training:
            categorical_fields, numerical_fields = self._sanetize_fields(categorical_fields, numerical_fields)
            parameters = self.compute_parameters(df, target_field, numerical_fields, categorical_fields, is_regression)

            self.set_state(TabularParametersState(parameters))
            self.set_state(ClassificationState(parameters["classes"]))
        else:
            parameters_state = self.get_state(TabularParametersState)
            parameters = parameters or (parameters_state.parameters if parameters_state is not None else None)
            if parameters is None:
                raise MisconfigurationException(
                    "Loading tabular data for evaluation or inference requires parameters from the train data. Either "
                    "construct the train data at the same time as evaluation and inference or provide the train "
                    "`datamodule.parameters` in the `parameters` argument."
                )

        self.parameters = parameters
        self.num_classes = len(parameters["classes"])

        # impute and normalize data
        df = _pre_transform(
            df,
            parameters["numerical_fields"],
            parameters["categorical_fields"],
            parameters["codes"],
            parameters["mean"],
            parameters["std"],
            parameters["target_field"],
            parameters["target_codes"],
        )

        cat_vars = _to_cat_vars_numpy(df, parameters["categorical_fields"])
        num_vars = _to_num_vars_numpy(df, parameters["numerical_fields"])

        num_samples = len(df)
        cat_vars = np.stack(cat_vars, 1) if len(cat_vars) else np.zeros((num_samples, 0))
        num_vars = np.stack(num_vars, 1) if len(num_vars) else np.zeros((num_samples, 0))

        if self.predicting:
            return [{DataKeys.INPUT: (c, n)} for c, n in zip(cat_vars, num_vars)]
        else:
            target = (
                df[parameters["target_field"]]
                .to_numpy()
                .astype(np.float32 if parameters["is_regression"] else np.int64)
            )
            return [{DataKeys.INPUT: (c, n), DataKeys.TARGET: t} for c, n, t in zip(cat_vars, num_vars, target)]


class TabularCSVInput(TabularDataFrameInput):
    def load_data(
        self,
        file: Optional[str],
        categorical_fields: Optional[List[str]] = None,
        numerical_fields: Optional[List[str]] = None,
        target_field: Optional[str] = None,
        parameters: Dict[str, Any] = None,
        is_regression: bool = True,
    ):
        if file is not None:
            return super().load_data(
                read_csv(file), categorical_fields, numerical_fields, target_field, is_regression, parameters
            )


class TabularDeserializer(Deserializer):
    @property
    def parameters(self) -> Dict[str, Any]:
        parameters_state = self.get_state(TabularParametersState)
        if parameters_state is None or parameters_state.parameters is None:
            raise MisconfigurationException(
                "Tabular tasks must previously have been trained in order to support serving as parameters from the "
                "train data are required."
            )
        return parameters_state.parameters

    def serve_load_sample(self, data: str) -> Any:
        parameters = self.parameters

        df = read_csv(StringIO(data))
        df = _pre_transform(
            df,
            parameters["numerical_fields"],
            parameters["categorical_fields"],
            parameters["codes"],
            parameters["mean"],
            parameters["std"],
            parameters["target_field"],
            parameters["target_codes"],
        )

        cat_vars = _to_cat_vars_numpy(df, parameters["categorical_fields"])
        num_vars = _to_num_vars_numpy(df, parameters["numerical_fields"])

        cat_vars = np.stack(cat_vars, 1)
        num_vars = np.stack(num_vars, 1)

        return [{DataKeys.INPUT: [c, n]} for c, n in zip(cat_vars, num_vars)]

    @property
    def example_input(self) -> str:
        parameters = self.parameters

        row = {}
        for cat_col in parameters["categorical_fields"]:
            row[cat_col] = ["test"]
        for num_col in parameters["numerical_fields"]:
            row[num_col] = [0]
        return str(DataFrame.from_dict(row).to_csv())


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


class TabularOutputTransform(OutputTransform):
    def uncollate(self, batch: Any) -> Any:
        return batch


class TabularData(DataModule):
    """Data module for tabular tasks."""

    input_transform_cls = TabularInputTransform
    output_transform_cls = TabularOutputTransform

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
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        **data_module_kwargs: Any,
    ) -> "TabularData":

        data_pipeline_state = DataPipelineState()

        train_input = TabularDataFrameInput(
            RunningStage.TRAINING,
            train_data_frame,
            categorical_fields=categorical_fields,
            numerical_fields=numerical_fields,
            target_field=target_fields,
            is_regression=cls.is_regression,
            data_pipeline_state=data_pipeline_state,
        )

        dataset_kwargs = dict(
            data_pipeline_state=data_pipeline_state, parameters=train_input.parameters if train_input else parameters
        )

        return cls(
            train_input,
            TabularDataFrameInput(RunningStage.VALIDATING, val_data_frame, **dataset_kwargs),
            TabularDataFrameInput(RunningStage.TESTING, test_data_frame, **dataset_kwargs),
            TabularDataFrameInput(RunningStage.PREDICTING, predict_data_frame, **dataset_kwargs),
            input_transform=cls.input_transform_cls(train_transform, val_transform, test_transform, predict_transform),
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
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        **data_module_kwargs: Any,
    ) -> "TabularData":

        data_pipeline_state = DataPipelineState()

        train_input = TabularCSVInput(
            RunningStage.TRAINING,
            train_file,
            categorical_fields=categorical_fields,
            numerical_fields=numerical_fields,
            target_field=target_fields,
            is_regression=cls.is_regression,
            data_pipeline_state=data_pipeline_state,
        )

        dataset_kwargs = dict(
            data_pipeline_state=data_pipeline_state, parameters=train_input.parameters if train_input else parameters
        )

        return cls(
            train_input,
            TabularCSVInput(RunningStage.VALIDATING, val_file, **dataset_kwargs),
            TabularCSVInput(RunningStage.TESTING, test_file, **dataset_kwargs),
            TabularCSVInput(RunningStage.PREDICTING, predict_file, **dataset_kwargs),
            input_transform=cls.input_transform_cls(train_transform, val_transform, test_transform, predict_transform),
            **data_module_kwargs,
        )
