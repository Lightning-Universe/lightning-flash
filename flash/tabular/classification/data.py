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
from typing import Any, Callable, Dict, List, Optional

from flash.core.data.process import Deserializer
from flash.core.utilities.imports import _PANDAS_AVAILABLE

if _PANDAS_AVAILABLE:
    from pandas.core.frame import DataFrame
else:
    DataFrame = object

from flash.tabular.data import TabularData, TabularPreprocess, TabularDataFrameDataSource


class TabularClassificationDataFrameDataSource(TabularDataFrameDataSource):
    def __init__(
            self,
            cat_cols: Optional[List[str]] = None,
            num_cols: Optional[List[str]] = None,
            target_col: Optional[str] = None,
            mean: Optional[DataFrame] = None,
            std: Optional[DataFrame] = None,
            codes: Optional[Dict[str, Any]] = None,
            target_codes: Optional[Dict[str, Any]] = None,
            classes: Optional[List[str]] = None
    ):
        super(TabularClassificationDataFrameDataSource, self).__init__(
            cat_cols=cat_cols,
            num_cols=num_cols,
            target_col=target_col,
            mean=mean,
            std=std,
            codes=codes,
            target_codes=target_codes,
            classes=classes,
            is_regression=False
        )


class TabularClassificationPreprocess(TabularPreprocess):

    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        cat_cols: Optional[List[str]] = None,
        num_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        mean: Optional[DataFrame] = None,
        std: Optional[DataFrame] = None,
        codes: Optional[Dict[str, Any]] = None,
        target_codes: Optional[Dict[str, Any]] = None,
        classes: Optional[List[str]] = None,
        deserializer: Optional[Deserializer] = None
    ):
        super(TabularClassificationPreprocess, self).__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            cat_cols=cat_cols,
            num_cols=num_cols,
            target_col=target_col,
            mean=mean,
            std=std,
            codes=codes,
            target_codes=target_codes,
            classes=classes,
            is_regression=False,
            deserializer=deserializer
        )


class TabularClassificationData(TabularData):
    is_regression = False
