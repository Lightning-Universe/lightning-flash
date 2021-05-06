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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from flash.core.classification import LabelsState
from flash.data.data_module import DataModule
from flash.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources
from flash.data.process import Preprocess
from flash.tabular.classification.data.dataset import (
    _compute_normalization,
    _generate_codes,
    _pre_transform,
    _to_cat_vars_numpy,
    _to_num_vars_numpy,
)


class TabularDataFrameDataSource(DataSource[DataFrame]):

    def __init__(
        self,
        cat_cols: Optional[List[str]] = None,
        num_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        mean: Optional[DataFrame] = None,
        std: Optional[DataFrame] = None,
        codes: Optional[Dict[str, Any]] = None,
        target_codes: Optional[Dict[str, Any]] = None,
        classes: Optional[List[str]] = None,
        is_regression: bool = True,
    ):
        super().__init__()

        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.target_col = target_col
        self.mean = mean
        self.std = std
        self.codes = codes
        self.target_codes = target_codes
        self.is_regression = is_regression

        self.set_state(LabelsState(classes))
        self.num_classes = len(classes)

    def common_load_data(
        self,
        df: DataFrame,
        dataset: Optional[Any] = None,
    ):
        # impute_data
        # compute train dataset stats
        dfs = _pre_transform([df], self.num_cols, self.cat_cols, self.codes, self.mean, self.std, self.target_col,
                             self.target_codes)

        df = dfs[0]

        if dataset is not None:
            dataset.num_samples = len(df)

        cat_vars = _to_cat_vars_numpy(df, self.cat_cols)
        num_vars = _to_num_vars_numpy(df, self.num_cols)

        cat_vars = np.stack(cat_vars, 1)  # if len(cat_vars) else np.zeros((len(self), 0))
        num_vars = np.stack(num_vars, 1)  # if len(num_vars) else np.zeros((len(self), 0))
        return df, cat_vars, num_vars

    def load_data(self, data: DataFrame, dataset: Optional[Any] = None):
        df, cat_vars, num_vars = self.common_load_data(data, dataset=dataset)
        target = df[self.target_col].to_numpy().astype(np.float32 if self.is_regression else np.int64)
        return [{
            DefaultDataKeys.INPUT: (c, n),
            DefaultDataKeys.TARGET: t
        } for c, n, t in zip(cat_vars, num_vars, target)]

    def predict_load_data(self, data: DataFrame, dataset: Optional[Any] = None):
        _, cat_vars, num_vars = self.common_load_data(data, dataset=dataset)
        return [{DefaultDataKeys.INPUT: (c, n)} for c, n in zip(cat_vars, num_vars)]


class TabularCSVDataSource(TabularDataFrameDataSource):

    def load_data(self, data: str, dataset: Optional[Any] = None):
        return super().load_data(pd.read_csv(data), dataset=dataset)

    def predict_load_data(self, data: str, dataset: Optional[Any] = None):
        return super().predict_load_data(pd.read_csv(data), dataset=dataset)


class TabularPreprocess(Preprocess):

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
        is_regression: bool = True,
    ):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.target_col = target_col
        self.mean = mean
        self.std = std
        self.codes = codes
        self.target_codes = target_codes
        self.classes = classes
        self.is_regression = is_regression

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources={
                DefaultDataSources.CSV: TabularCSVDataSource(
                    cat_cols, num_cols, target_col, mean, std, codes, target_codes, classes, is_regression
                ),
                "df": TabularDataFrameDataSource(
                    cat_cols, num_cols, target_col, mean, std, codes, target_codes, classes, is_regression
                ),
            },
            default_data_source=DefaultDataSources.CSV,
        )

    def get_state_dict(self, strict: bool = False) -> Dict[str, Any]:
        return {
            "cat_cols": self.cat_cols,
            "num_cols": self.num_cols,
            "target_col": self.target_col,
            "mean": self.mean,
            "std": self.std,
            "codes": self.codes,
            "target_codes": self.target_codes,
            "classes": self.classes,
            "is_regression": self.is_regression,
        }

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = True) -> 'Preprocess':
        return cls(**state_dict)


class TabularData(DataModule):
    """Data module for tabular tasks"""

    preprocess_cls = TabularPreprocess

    @property
    def codes(self) -> Dict[str, str]:
        return self._data_source.codes

    @property
    def num_classes(self) -> int:
        return self._data_source.num_classes

    @property
    def cat_cols(self) -> Optional[List[str]]:
        return self._data_source.cat_cols

    @property
    def num_cols(self) -> Optional[List[str]]:
        return self._data_source.num_cols

    @property
    def num_features(self) -> int:
        return len(self.cat_cols) + len(self.num_cols)

    @property
    def emb_sizes(self) -> list:
        """Recommended embedding sizes."""

        # https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
        # The following "formula" provides a general rule of thumb about the number of embedding dimensions:
        # embedding_dimensions =  number_of_categories**0.25
        num_classes = [len(self.codes[cat]) for cat in self.cat_cols]
        emb_dims = [max(int(n**0.25), 16) for n in num_classes]
        return list(zip(num_classes, emb_dims))

    @staticmethod
    def _sanetize_cols(cat_cols: Optional[List], num_cols: Optional[List]):
        if cat_cols is None and num_cols is None:
            raise RuntimeError('Both `cat_cols` and `num_cols` are None!')

        return cat_cols or [], num_cols or []

    @classmethod
    def compute_state(
        cls,
        train_df: DataFrame,
        val_df: Optional[DataFrame],
        test_df: Optional[DataFrame],
        predict_df: Optional[DataFrame],
        target_col: str,
        num_cols: List[str],
        cat_cols: List[str],
    ) -> Tuple[float, float, List[str], Dict[str, Any], Dict[str, Any]]:

        if train_df is None:
            raise MisconfigurationException("train_df is required to instantiate the TabularDataFrameDataSource")

        dfs = [train_df]

        if val_df is not None:
            dfs += [val_df]

        if test_df is not None:
            dfs += [test_df]

        if predict_df is not None:
            dfs += [predict_df]

        mean, std = _compute_normalization(dfs[0], num_cols)
        classes = list(dfs[0][target_col].unique())

        if dfs[0][target_col].dtype == object:
            # if the target_col is a category, not an int
            target_codes = _generate_codes(dfs, [target_col])
        else:
            target_codes = None
        codes = _generate_codes(dfs, cat_cols)

        return mean, std, classes, codes, target_codes

    @classmethod
    def from_df(
        cls,
        categorical_cols: List,
        numerical_cols: List,
        target_col: str,
        train_df: DataFrame,
        val_df: Optional[DataFrame] = None,
        test_df: Optional[DataFrame] = None,
        predict_df: Optional[DataFrame] = None,
        is_regression: bool = False,
        preprocess: Optional[Preprocess] = None,
        val_split: float = None,
        batch_size: int = 8,
        num_workers: Optional[int] = None,
    ):
        """Creates a TabularData object from pandas DataFrames.

        Args:
            train_df: Train data DataFrame.
            target_col: The column containing the class id.
            categorical_cols: The list of categorical columns.
            numerical_cols: The list of numerical columns.
            val_df: Validation data DataFrame.
            test_df: Test data DataFrame.
            batch_size: The batchsize to use for parallel loading. Defaults to 64.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to None which equals the number of available CPU threads,
                or 0 for Darwin platform.
            val_split: Float between 0 and 1 to create a validation dataset from train dataset.
            preprocess: Preprocess to be used within this DataModule DataPipeline.

        Returns:
            TabularData: The constructed data module.

        Examples::

            text_data = TextClassificationData.from_files("train.csv", label_field="class", text_field="sentence")
        """
        categorical_cols, numerical_cols = cls._sanetize_cols(categorical_cols, numerical_cols)

        mean, std, classes, codes, target_codes = cls.compute_state(
            train_df, val_df, test_df, predict_df, target_col, numerical_cols, categorical_cols
        )

        return cls.from_data_source(
            data_source="df",
            train_data=train_df,
            val_data=val_df,
            test_data=test_df,
            predict_data=predict_df,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            cat_cols=categorical_cols,
            num_cols=numerical_cols,
            target_col=target_col,
            mean=mean,
            std=std,
            codes=codes,
            target_codes=target_codes,
            classes=classes,
            is_regression=is_regression,
        )

    @classmethod
    def from_csv(
        cls,
        categorical_fields: Union[str, List[str]],
        numerical_fields: Union[str, List[str]],
        target_field: Optional[str] = None,
        train_file: Optional[str] = None,
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        predict_file: Optional[str] = None,
        is_regression: bool = False,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
    ) -> 'DataModule':
        return cls.from_df(
            categorical_fields,
            numerical_fields,
            target_field,
            train_df=pd.read_csv(train_file) if train_file is not None else None,
            val_df=pd.read_csv(val_file) if val_file is not None else None,
            test_df=pd.read_csv(test_file) if test_file is not None else None,
            predict_df=pd.read_csv(predict_file) if predict_file is not None else None,
            is_regression=is_regression,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
        )
