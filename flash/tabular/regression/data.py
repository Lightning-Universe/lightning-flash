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
from typing import Any, Callable, Dict, List, Optional, Union

from flash.core.data.callback import BaseDataFetcher
from flash.core.data.process import Deserializer
from flash.core.utilities.imports import _PANDAS_AVAILABLE

if _PANDAS_AVAILABLE:
    from pandas.core.frame import DataFrame
else:
    DataFrame = object

from pytorch_forecasting import TimeSeriesDataSet

from flash.tabular.data import TabularData, TabularDataFrameDataSource, TabularPreprocess


class TabularRegressionDataFrameDataSource(TabularDataFrameDataSource):

    def __init__(
        self,
        time_idx: str,
        target: Union[str, List[str]],
        group_ids: List[str],
        cat_cols: Optional[List[str]] = None,
        num_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        mean: Optional[DataFrame] = None,
        std: Optional[DataFrame] = None,
        codes: Optional[Dict[str, Any]] = None,
        target_codes: Optional[Dict[str, Any]] = None,
        classes: Optional[List[str]] = None,
        **data_source_kwargs: Any
    ):
        self.time_idx = time_idx
        self.target = target
        self.group_ids = group_ids
        self.data_source_kwargs = data_source_kwargs
        super(TabularRegressionDataFrameDataSource, self).__init__(
            cat_cols=cat_cols,
            num_cols=num_cols,
            target_col=target_col,
            mean=mean,
            std=std,
            codes=codes,
            target_codes=target_codes,
            classes=classes,
            is_regression=True
        )

    def load_data(self, data: DataFrame, dataset: Optional[Any] = None):
        return TimeSeriesDataSet(
            data, time_idx=self.time_idx, group_ids=self.group_ids, target=self.target, **self.data_source_kwargs
        )


class TabularRegressionPreprocess(TabularPreprocess):

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
        deserializer: Optional[Deserializer] = None,
        **data_source_kwargs: Any
    ):
        super(TabularRegressionPreprocess, self).__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources={
                "data_frame": TabularRegressionDataFrameDataSource(
                    cat_cols,
                    num_cols,
                    target_col,
                    mean,
                    std,
                    codes,
                    target_codes,
                    classes,
                    is_regression=True,
                    **data_source_kwargs
                ),
            },
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


class TabularRegressionData(TabularData):
    is_regression = True
    preprocess_cls = TabularRegressionPreprocess

    @classmethod
    def from_data_frame(
        cls,
        group_ids: Optional[List[str]] = None,
        target: Optional[str] = None,
        time_idx: Optional[str] = None,
        categorical_fields: Optional[Union[str, List[str]]] = None,
        numerical_fields: Optional[Union[str, List[str]]] = None,
        target_fields: Optional[str] = None,
        train_data_frame: Optional[DataFrame] = None,
        val_data_frame: Optional[DataFrame] = None,
        test_data_frame: Optional[DataFrame] = None,
        predict_data_frame: Optional[DataFrame] = None,
        min_encoder_length: Optional[int] = None,
        max_encoder_length: Optional[int] = None,
        min_prediction_length: Optional[int] = None,
        max_prediction_length: Optional[int] = None,
        time_varying_unknown_reals: Optional[List[str]] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        preprocess: Optional[TabularRegressionPreprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = None,
        num_workers: Optional[int] = None,
        **preprocess_kwargs: Any,
    ):
        super().from_data_frame(
            time_idx=time_idx,
            group_ids=group_ids,
            target=target,
            categorical_fields=categorical_fields,
            numerical_fields=numerical_fields,
            target_fields=target_fields,
            train_data_frame=train_data_frame,
            val_data_frame=val_data_frame,
            test_data_frame=test_data_frame,
            predict_data_frame=predict_data_frame,
            time_varying_unknown_reals=time_varying_unknown_reals,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            **preprocess_kwargs
        )
