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
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
from torch import Tensor

from flash.core.classification import ClassificationDataPipeline
from flash.core.data import DataPipeline
from flash.core.data.datamodule import DataModule
from flash.core.data.utils import _contains_any_tensor
from flash.tabular.classification.data.dataset import (
    _compute_normalization,
    _dfs_to_samples,
    _generate_codes,
    _impute,
    _pre_transform,
    PandasDataset,
)


class TabularDataPipeline(ClassificationDataPipeline):

    def __init__(
        self,
        categorical_input: List,
        numerical_input: List,
        target: str,
        mean: DataFrame,
        std: DataFrame,
        codes: Dict,
    ):
        self._categorical_input = categorical_input
        self._numerical_input = numerical_input
        self._target = target
        self._mean = mean
        self._std = std
        self._codes = codes

    def before_collate(self, samples: Any) -> Any:
        """Override to apply transformations to samples"""
        if _contains_any_tensor(samples, dtype=(Tensor, np.ndarray)):
            return samples
        if isinstance(samples, str):
            samples = pd.read_csv(samples)
        if isinstance(samples, DataFrame):
            samples = [samples]
        dfs = _pre_transform(
            samples, self._numerical_input, self._categorical_input, self._codes, self._mean, self._std
        )
        return _dfs_to_samples(dfs, self._categorical_input, self._numerical_input)


class TabularData(DataModule):
    """Data module for tabular tasks"""

    def __init__(
        self,
        train_df: DataFrame,
        target: str,
        categorical_input: Optional[List] = None,
        numerical_input: Optional[List] = None,
        valid_df: Optional[DataFrame] = None,
        test_df: Optional[DataFrame] = None,
        batch_size: int = 2,
        num_workers: Optional[int] = None,
    ):
        dfs = [train_df]
        self._test_df = None

        if categorical_input is None and numerical_input is None:
            raise RuntimeError('Both `categorical_input` and `numerical_input` are None!')

        categorical_input = categorical_input if categorical_input is not None else []
        numerical_input = numerical_input if numerical_input is not None else []

        if valid_df is not None:
            dfs.append(valid_df)

        if test_df is not None:
            # save for predict function
            self._test_df = test_df.copy()
            self._test_df.drop(target, axis=1)
            dfs.append(test_df)

        # impute missing values
        dfs = _impute(dfs, numerical_input)

        # compute train dataset stats
        self.mean, self.std = _compute_normalization(dfs[0], numerical_input)

        if dfs[0][target].dtype == object:
            # if the target is a category, not an int
            self.target_codes = _generate_codes(dfs, [target])
        else:
            self.target_codes = None

        self.codes = _generate_codes(dfs, categorical_input)

        dfs = _pre_transform(
            dfs, numerical_input, categorical_input, self.codes, self.mean, self.std, target, self.target_codes
        )

        # normalize
        self.cat_cols = categorical_input
        self.num_cols = numerical_input

        self._num_classes = len(train_df[target].unique())

        train_ds = PandasDataset(dfs[0], categorical_input, numerical_input, target)
        valid_ds = PandasDataset(dfs[1], categorical_input, numerical_input, target) if valid_df is not None else None
        test_ds = PandasDataset(dfs[-1], categorical_input, numerical_input, target) if test_df is not None else None
        super().__init__(train_ds, valid_ds, test_ds, batch_size=batch_size, num_workers=num_workers)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def num_features(self) -> int:
        return len(self.cat_cols) + len(self.num_cols)

    @classmethod
    def from_df(
        cls,
        train_df: DataFrame,
        target: str,
        categorical_input: Optional[List] = None,
        numerical_input: Optional[List] = None,
        valid_df: Optional[DataFrame] = None,
        test_df: Optional[DataFrame] = None,
        batch_size: int = 8,
        num_workers: Optional[int] = None,
        val_size: float = None,
        test_size: float = None,
    ):
        """Creates a TabularData object from pandas DataFrames.

        Args:
            train_df: train data DataFrame
            target: The column containing the class id.
            categorical_input: The list of categorical columns.
            numerical_input: The list of numerical columns.
            valid_df: validation data DataFrame
            test_df: test data DataFrame
            batch_size: the batchsize to use for parallel loading. Defaults to 64.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to None which equals the number of available CPU threads.
            val_size: float between 0 and 1 to create a validation dataset from train dataset
            test_size: float between 0 and 1 to create a test dataset from train validation

        Returns:
            TabularData: The constructed data module.

        Examples::

            text_data = TextClassificationData.from_files("train.csv", label_field="class", text_field="sentence")
        """
        if valid_df is None and isinstance(val_size, float) and isinstance(test_size, float):
            assert 0 < val_size and val_size < 1
            assert 0 < test_size and test_size < 1
            train_df, valid_df = train_test_split(train_df, test_size=(val_size + test_size))

        if test_df is None and isinstance(test_size, float):
            assert 0 < test_size and test_size < 1
            valid_df, test_df = train_test_split(valid_df, test_size=test_size)

        datamodule = cls(
            train_df=train_df,
            target=target,
            categorical_input=categorical_input,
            numerical_input=numerical_input,
            valid_df=valid_df,
            test_df=test_df,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        datamodule.data_pipeline = TabularDataPipeline(
            categorical_input, numerical_input, target, datamodule.mean, datamodule.std, datamodule.codes
        )

        return datamodule

    @classmethod
    def from_csv(
        cls,
        train_csv: str,
        target: str,
        categorical_input: Optional[List] = None,
        numerical_input: Optional[List] = None,
        valid_csv: Optional[str] = None,
        test_csv: Optional[str] = None,
        batch_size: int = 8,
        num_workers: Optional[int] = None,
        val_size: Optional[float] = None,
        test_size: Optional[float] = None,
        **pandas_kwargs,
    ):
        """Creates a TextClassificationData object from pandas DataFrames.

        Args:
            train_csv: train data csv file.
            target: The column containing the class id.
            categorical_input: The list of categorical columns.
            numerical_input: The list of numerical columns.
            valid_csv: validation data csv file.
            test_csv: test data csv file.
            batch_size: the batchsize to use for parallel loading. Defaults to 64.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to None which equals the number of available CPU threads.
            val_size: float between 0 and 1 to create a validation dataset from train dataset
            test_size: float between 0 and 1 to create a test dataset from train validation

        Returns:
            TabularData: The constructed data module.

        Examples::

            text_data = TabularData.from_files("train.csv", label_field="class", text_field="sentence")
        """
        train_df = pd.read_csv(train_csv, **pandas_kwargs)
        valid_df = pd.read_csv(valid_csv, **pandas_kwargs) if valid_csv is not None else None
        test_df = pd.read_csv(test_csv, **pandas_kwargs) if test_csv is not None else None
        datamodule = cls.from_df(
            train_df, target, categorical_input, numerical_input, valid_df, test_df, batch_size, num_workers, val_size,
            test_size
        )
        return datamodule

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
    def default_pipeline() -> DataPipeline():
        # TabularDataPipeline depends on the data
        return DataPipeline()
