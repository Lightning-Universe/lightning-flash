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
from io import StringIO
from typing import Any, Dict, List, Optional, Union

import numpy as np

from flash.core.data.io.input import DataKeys, Input, ServeInput
from flash.core.utilities.imports import _PANDAS_AVAILABLE
from flash.tabular.classification.utils import (
    _compute_normalization,
    _generate_codes,
    _pre_transform,
    _to_cat_vars_numpy,
    _to_num_vars_numpy,
)

if _PANDAS_AVAILABLE:
    import pandas as pd
    from pandas.core.frame import DataFrame
else:
    DataFrame = object


class TabularDataFrameInput(Input):
    parameters: dict

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
        numerical_fields: List[str],
        categorical_fields: List[str],
    ) -> Dict[str, Any]:

        mean, std = _compute_normalization(train_data_frame, numerical_fields)

        codes = _generate_codes(train_data_frame, categorical_fields)

        return dict(
            mean=mean,
            std=std,
            codes=codes,
            numerical_fields=numerical_fields,
            categorical_fields=categorical_fields,
        )

    def preprocess(
        self,
        df: DataFrame,
        categorical_fields: Optional[List[str]] = None,
        numerical_fields: Optional[List[str]] = None,
        parameters: Dict[str, Any] = None,
    ):
        if self.training:
            categorical_fields, numerical_fields = self._sanetize_fields(categorical_fields, numerical_fields)
            parameters = self.compute_parameters(df, numerical_fields, categorical_fields)
        elif parameters is None:
            raise ValueError(
                "Loading tabular data for evaluation or inference requires parameters from the train data. Either "
                "construct the train data at the same time as evaluation and inference or provide the train "
                "`datamodule.parameters` in the `parameters` argument."
            )

        self.parameters = parameters

        # impute and normalize data
        df = _pre_transform(
            df,
            parameters["numerical_fields"],
            parameters["categorical_fields"],
            parameters["codes"],
            parameters["mean"],
            parameters["std"],
        )

        cat_vars = _to_cat_vars_numpy(df, parameters["categorical_fields"])
        num_vars = _to_num_vars_numpy(df, parameters["numerical_fields"])

        num_samples = len(df)
        cat_vars = np.stack(cat_vars, 1) if len(cat_vars) else np.zeros((num_samples, 0), dtype=np.int64)
        num_vars = np.stack(num_vars, 1) if len(num_vars) else np.zeros((num_samples, 0), dtype=np.float32)

        return cat_vars, num_vars


class TabularDeserializer(ServeInput):
    def __init__(self, *args, parameters: Optional[Dict[str, Any]] = None, **kwargs):
        self._parameters = parameters
        super().__init__(*args, **kwargs)

    def serve_load_sample(self, data: str) -> Any:
        parameters = self._parameters

        df = pd.read_csv(StringIO(data))
        df = _pre_transform(
            df,
            parameters["numerical_fields"],
            parameters["categorical_fields"],
            parameters["codes"],
            parameters["mean"],
            parameters["std"],
        )

        cat_vars = _to_cat_vars_numpy(df, parameters["categorical_fields"])
        num_vars = _to_num_vars_numpy(df, parameters["numerical_fields"])

        cat_vars = np.stack(cat_vars, 1)
        num_vars = np.stack(num_vars, 1)

        return [{DataKeys.INPUT: [c, n]} for c, n in zip(cat_vars, num_vars)]

    @property
    def example_input(self) -> str:
        parameters = self._parameters

        row = {}
        for cat_col in parameters["categorical_fields"]:
            row[cat_col] = ["test"]
        for num_col in parameters["numerical_fields"]:
            row[num_col] = [0]
        return str(DataFrame.from_dict(row).to_csv())
