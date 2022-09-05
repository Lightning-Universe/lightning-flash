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
from typing import Any, Dict, List, Optional, Union

import numpy as np

from flash.core.data.io.input import DataKeys
from flash.core.data.utilities.loading import load_data_frame
from flash.core.utilities.imports import _PANDAS_AVAILABLE
from flash.tabular.input import TabularDataFrameInput

if _PANDAS_AVAILABLE:
    from pandas.core.frame import DataFrame
else:
    DataFrame = object


class TabularRegressionDataFrameInput(TabularDataFrameInput):
    def load_data(
        self,
        data_frame: DataFrame,
        categorical_fields: Optional[Union[str, List[str]]] = None,
        numerical_fields: Optional[Union[str, List[str]]] = None,
        target_field: Optional[str] = None,
        parameters: Dict[str, Any] = None,
    ):
        cat_vars, num_vars = self.preprocess(data_frame, categorical_fields, numerical_fields, parameters)

        if not self.predicting:
            targets = data_frame[target_field].to_numpy().astype(np.float32)
            return [{DataKeys.INPUT: (c, n), DataKeys.TARGET: t} for c, n, t in zip(cat_vars, num_vars, targets)]
        else:
            return [{DataKeys.INPUT: (c, n)} for c, n in zip(cat_vars, num_vars)]


class TabularRegressionCSVInput(TabularRegressionDataFrameInput):
    def load_data(
        self,
        file: Optional[str],
        categorical_fields: Optional[Union[str, List[str]]] = None,
        numerical_fields: Optional[Union[str, List[str]]] = None,
        target_field: Optional[str] = None,
        parameters: Dict[str, Any] = None,
    ):
        if file is not None:
            return super().load_data(
                load_data_frame(file), categorical_fields, numerical_fields, target_field, parameters
            )


class TabularRegressionDictInput(TabularRegressionDataFrameInput):
    def load_data(
        self,
        data: Dict[str, List[Any]],
        categorical_fields: Optional[Union[str, List[str]]] = None,
        numerical_fields: Optional[Union[str, List[str]]] = None,
        target_field: Optional[str] = None,
        parameters: Dict[str, Any] = None,
    ):
        data_frame = DataFrame.from_dict(data)

        return super().load_data(data_frame, categorical_fields, numerical_fields, target_field, parameters)


class TabularRegressionListInput(TabularRegressionDataFrameInput):
    def load_data(
        self,
        data: List[Union[tuple, dict]],
        categorical_fields: Optional[Union[str, List[str]]] = None,
        numerical_fields: Optional[Union[str, List[str]]] = None,
        target_field: Optional[str] = None,
        parameters: Dict[str, Any] = None,
    ):
        data_frame = DataFrame.from_records(data)

        return super().load_data(data_frame, categorical_fields, numerical_fields, target_field, parameters)
