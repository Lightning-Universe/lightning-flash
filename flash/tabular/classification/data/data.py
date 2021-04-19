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
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from sklearn.model_selection import train_test_split

from flash.core.classification import ClassificationState
from flash.data.auto_dataset import AutoDataset
from flash.data.data_module import DataModule
from flash.data.process import Preprocess
from flash.tabular.classification.data.dataset import (
    _compute_normalization,
    _generate_codes,
    _pre_transform,
    _to_cat_vars_numpy,
    _to_num_vars_numpy,
)


class TabularPreprocess(Preprocess):

    def __init__(
        self,
        train_df: DataFrame,
        val_df: Optional[DataFrame],
        test_df: Optional[DataFrame],
        predict_df: Optional[DataFrame],
        target_col: str,
        num_cols: List[str],
        cat_cols: List[str],
        is_regression: bool,
    ):
        super().__init__()
        if train_df is None:
            raise MisconfigurationException("train_df is required to instantiate the TabularPreprocess")

        dfs = [train_df]

        if val_df is not None:
            dfs += [val_df]

        if test_df is not None:
            dfs += [test_df]

        if predict_df is not None:
            dfs += [predict_df]

        mean, std = _compute_normalization(dfs[0], num_cols)
        classes = dfs[0][target_col].unique()
        self.set_state(ClassificationState(classes))
        num_classes = len(classes)
        if dfs[0][target_col].dtype == object:
            # if the target_col is a category, not an int
            target_codes = _generate_codes(dfs, [target_col])
        else:
            target_codes = None
        codes = _generate_codes(dfs, cat_cols)

        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.target_col = target_col
        self.mean = mean
        self.std = std
        self.codes = codes
        self.target_codes = target_codes
        self.num_classes = num_classes
        self.is_regression = is_regression

    def common_load_data(self, df: DataFrame, dataset: AutoDataset):
        # impute_data
        # compute train dataset stats
        dfs = _pre_transform([df], self.num_cols, self.cat_cols, self.codes, self.mean, self.std, self.target_col,
                             self.target_codes)

        df = dfs[0]

        dataset.num_samples = len(df)
        cat_vars = _to_cat_vars_numpy(df, self.cat_cols)
        num_vars = _to_num_vars_numpy(df, self.num_cols)

        cat_vars = np.stack(cat_vars, 1) if len(cat_vars) else np.zeros((len(self), 0))
        num_vars = np.stack(num_vars, 1) if len(num_vars) else np.zeros((len(self), 0))
        return df, cat_vars, num_vars

    def load_data(self, df: DataFrame, dataset: AutoDataset):
        df, cat_vars, num_vars = self.common_load_data(df, dataset)
        target = df[self.target_col].to_numpy().astype(np.float32 if self.is_regression else np.int64)
        return [((c, n), t) for c, n, t in zip(cat_vars, num_vars, target)]

    def predict_load_data(self, sample: Union[str, DataFrame], dataset: AutoDataset):
        df = pd.read_csv(sample) if isinstance(sample, str) else sample
        _, cat_vars, num_vars = self.common_load_data(df, dataset)
        return list(zip(cat_vars, num_vars))


class TabularData(DataModule):
    """Data module for tabular tasks"""

    _preprocess: Preprocess = None

    @property
    def codes(self) -> Dict[str, str]:
        return self._preprocess.codes

    @property
    def num_classes(self) -> int:
        return self._preprocess.num_classes

    @property
    def cat_cols(self) -> Optional[List[str]]:
        return self._preprocess.cat_cols

    @property
    def num_cols(self) -> Optional[List[str]]:
        return self._preprocess.num_cols

    @property
    def num_features(self) -> int:
        return len(self.cat_cols) + len(self.num_cols)

    @classmethod
    def from_csv(
        cls,
        target_col: str,
        train_csv: Optional[str] = None,
        categorical_cols: Optional[List] = None,
        numerical_cols: Optional[List] = None,
        val_csv: Optional[str] = None,
        test_csv: Optional[str] = None,
        predict_csv: Optional[str] = None,
        batch_size: int = 8,
        num_workers: Optional[int] = None,
        val_size: Optional[float] = None,
        test_size: Optional[float] = None,
        preprocess: Optional[Preprocess] = None,
        **pandas_kwargs,
    ):
        """Creates a TextClassificationData object from pandas DataFrames.

        Args:
            train_csv: Train data csv file.
            target_col: The column containing the class id.
            categorical_cols: The list of categorical columns.
            numerical_cols: The list of numerical columns.
            val_csv: Validation data csv file.
            test_csv: Test data csv file.
            batch_size: The batchsize to use for parallel loading. Defaults to 64.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to None which equals the number of available CPU threads,
                or 0 for Darwin platform.
            val_size: Float between 0 and 1 to create a validation dataset from train dataset.
            test_size: Float between 0 and 1 to create a test dataset from train validation.
            preprocess: Preprocess to be used within this DataModule DataPipeline.

        Returns:
            TabularData: The constructed data module.

        Examples::

            text_data = TabularData.from_files("train.csv", label_field="class", text_field="sentence")
        """
        train_df = pd.read_csv(train_csv, **pandas_kwargs)
        val_df = pd.read_csv(val_csv, **pandas_kwargs) if val_csv else None
        test_df = pd.read_csv(test_csv, **pandas_kwargs) if test_csv else None
        predict_df = pd.read_csv(predict_csv, **pandas_kwargs) if predict_csv else None

        return cls.from_df(
            train_df,
            target_col,
            categorical_cols,
            numerical_cols,
            val_df,
            test_df,
            predict_df,
            batch_size,
            num_workers,
            val_size,
            test_size,
            preprocess=preprocess,
        )

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
    def _split_dataframe(
        train_df: DataFrame,
        val_df: Optional[DataFrame] = None,
        test_df: Optional[DataFrame] = None,
        val_size: float = None,
        test_size: float = None,
    ):
        if val_df is None and isinstance(val_size, float) and isinstance(test_size, float):
            assert 0 < val_size < 1
            assert 0 < test_size < 1
            train_df, val_df = train_test_split(train_df, test_size=(val_size + test_size))

        if test_df is None and isinstance(test_size, float):
            assert 0 < test_size < 1
            val_df, test_df = train_test_split(val_df, test_size=test_size)

        return train_df, val_df, test_df

    @staticmethod
    def _sanetize_cols(cat_cols: Optional[List], num_cols: Optional[List]):
        if cat_cols is None and num_cols is None:
            raise RuntimeError('Both `cat_cols` and `num_cols` are None!')

        return cat_cols or [], num_cols or []

    @classmethod
    def from_df(
        cls,
        train_df: DataFrame,
        target_col: str,
        categorical_cols: Optional[List] = None,
        numerical_cols: Optional[List] = None,
        val_df: Optional[DataFrame] = None,
        test_df: Optional[DataFrame] = None,
        predict_df: Optional[DataFrame] = None,
        batch_size: int = 8,
        num_workers: Optional[int] = None,
        val_size: float = None,
        test_size: float = None,
        is_regression: bool = False,
        preprocess: Optional[Preprocess] = None,
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
            val_size: Float between 0 and 1 to create a validation dataset from train dataset.
            test_size: Float between 0 and 1 to create a test dataset from train validation.
            preprocess: Preprocess to be used within this DataModule DataPipeline.

        Returns:
            TabularData: The constructed data module.

        Examples::

            text_data = TextClassificationData.from_files("train.csv", label_field="class", text_field="sentence")
        """
        categorical_cols, numerical_cols = cls._sanetize_cols(categorical_cols, numerical_cols)

        train_df, val_df, test_df = cls._split_dataframe(train_df, val_df, test_df, val_size, test_size)

        preprocess = preprocess or TabularPreprocess(
            train_df,
            val_df,
            test_df,
            predict_df,
            target_col,
            numerical_cols,
            categorical_cols,
            is_regression,
        )

        return cls.from_load_data_inputs(
            train_load_data_input=train_df,
            val_load_data_input=val_df,
            test_load_data_input=test_df,
            predict_load_data_input=predict_df,
            batch_size=batch_size,
            num_workers=num_workers,
            preprocess=preprocess
        )
