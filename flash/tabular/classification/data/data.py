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
from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from sklearn.model_selection import train_test_split

from flash.data.auto_dataset import AutoDataset
from flash.data.data_module import DataModule
from flash.data.process import Preprocess, PreprocessState
from flash.tabular.classification.data.dataset import (
    _compute_normalization,
    _dfs_to_samples,
    _generate_codes,
    _impute,
    _pre_transform,
    _to_cat_vars_numpy,
    _to_num_cols_numpy,
    PandasDataset,
)


@dataclass(unsafe_hash=True, frozen=True)
class TabularState(PreprocessState):
    cat_cols: List[str]
    num_cols: List[str]
    target: str
    mean: DataFrame
    std: DataFrame
    codes: Dict
    target_codes: Dict
    num_classes: int
    regression: bool


class TabularPreprocess(Preprocess):

    def __init__(
        self,
        cat_cols: List,
        num_cols: List,
        target: str,
        mean: DataFrame,
        std: DataFrame,
        codes: Dict,
        target_codes: Dict,
        num_classes: int,
        regression: bool = False,
    ):
        super().__init__()
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.target = target
        self.mean = mean
        self.std = std
        self.codes = codes
        self.target_codes = target_codes
        self.num_classes = num_classes
        self.regression = regression

    @property
    def state(self) -> TabularState:
        return TabularState(
            self.cat_cols, self.num_cols, self.target, self.mean, self.std, self.codes, self.target_codes,
            self.num_classes, self.regression
        )

    @staticmethod
    def generate_state(
        train_df: DataFrame,
        valid_df: Optional[DataFrame],
        test_df: Optional[DataFrame],
        predict_df: Optional[DataFrame],
        target: str,
        num_cols: List,
        cat_cols: List,
        regression: bool,
        preprocess_state: Optional[TabularState] = None
    ):
        if preprocess_state is not None:
            return preprocess_state

        if train_df is None:
            raise MisconfigurationException("train_df is required to compute the preprocess state")

        dfs = [train_df]

        if valid_df is not None:
            dfs += [valid_df]

        if test_df is not None:
            dfs += [test_df]

        if predict_df is not None:
            dfs += [predict_df]

        mean, std = _compute_normalization(dfs[0], num_cols)
        codes = _generate_codes(dfs, [target])
        num_classes = len(dfs[0][target].unique())
        if dfs[0][target].dtype == object:
            # if the target is a category, not an int
            target_codes = _generate_codes(dfs, [target])
        else:
            target_codes = None
        codes = _generate_codes(dfs, cat_cols)

        return TabularState(
            cat_cols,
            num_cols,
            target,
            mean,
            std,
            codes,
            target_codes,
            num_classes,
            regression,
        )

    def common_load_data(self, df: DataFrame, dataset: AutoDataset):
        # impute_data
        dfs = _impute([df], self.num_cols)

        # compute train dataset stats
        dfs = _pre_transform(
            dfs, self.num_cols, self.cat_cols, self.codes, self.mean, self.std, self.target, self.target_codes
        )

        df = dfs[0]

        dataset.num_samples = len(df)
        cat_vars = _to_cat_vars_numpy(df, self.cat_cols)
        num_vars = _to_num_cols_numpy(df, self.num_cols)
        dataset.num_samples = len(df)
        cat_vars = np.stack(cat_vars, 1) if len(cat_vars) else np.zeros((len(self), 0))
        num_vars = np.stack(num_vars, 1) if len(num_vars) else np.zeros((len(self), 0))
        return df, cat_vars, num_vars

    def load_data(self, df: DataFrame, dataset: AutoDataset):
        df, cat_vars, num_vars = self.common_load_data(df, dataset)
        target = df[self.target].to_numpy().astype(np.float32 if self.regression else np.int64)
        return [((c, n), t) for c, n, t in zip(cat_vars, num_vars, target)]

    def predict_load_data(self, sample: Union[str, DataFrame], dataset: AutoDataset):
        df = pd.read_csv(sample) if isinstance(sample, str) else sample
        _, cat_vars, num_vars = self.common_load_data(df, dataset)
        return [(c, n) for c, n in zip(cat_vars, num_vars)]


class TabularData(DataModule):
    """Data module for tabular tasks"""

    preprocess_cls = TabularPreprocess

    @property
    def preprocess_state(self):
        return self._preprocess.state

    @preprocess_state.setter
    def preprocess_state(self, preprocess_state):
        self._preprocess = self.preprocess_cls.from_state(preprocess_state)

    @property
    def codes(self):
        return self.preprocess_state.codes

    @property
    def num_classes(self) -> int:
        return self.preprocess_state.num_classes

    @property
    def cat_cols(self):
        return self.preprocess_state.cat_cols

    @property
    def num_cols(self):
        return self.preprocess_state.num_cols

    @property
    def num_features(self) -> int:
        return len(self.cat_cols) + len(self.num_cols)

    @classmethod
    def from_csv(
        cls,
        target: str,
        train_csv: Optional[str] = None,
        cat_cols: Optional[List] = None,
        num_cols: Optional[List] = None,
        valid_csv: Optional[str] = None,
        test_csv: Optional[str] = None,
        predict_csv: Optional[str] = None,
        batch_size: int = 8,
        num_workers: Optional[int] = None,
        val_size: Optional[float] = None,
        test_size: Optional[float] = None,
        preprocess_cls: Optional[Type[Preprocess]] = None,
        preprocess_state: Optional[TabularState] = None,
        **pandas_kwargs,
    ):
        """Creates a TextClassificationData object from pandas DataFrames.

        Args:
            train_csv: train data csv file.
            target: The column containing the class id.
            cat_cols: The list of categorical columns.
            num_cols: The list of numerical columns.
            valid_csv: validation data csv file.
            test_csv: test data csv file.
            batch_size: the batchsize to use for parallel loading. Defaults to 64.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to None which equals the number of available CPU threads,
            or 0 for Darwin platform.
            val_size: float between 0 and 1 to create a validation dataset from train dataset
            test_size: float between 0 and 1 to create a test dataset from train validation
            preprocess_cls: Preprocess class to be used within this DataModule DataPipeline
            preprocess_state: Used to store the train statistics

        Returns:
            TabularData: The constructed data module.

        Examples::

            text_data = TabularData.from_files("train.csv", label_field="class", text_field="sentence")
        """
        train_df = pd.read_csv(train_csv, **pandas_kwargs)
        valid_df = pd.read_csv(valid_csv, **pandas_kwargs) if valid_csv is not None else None
        test_df = pd.read_csv(test_csv, **pandas_kwargs) if test_csv is not None else None
        predict_df = pd.read_csv(predict_csv, **pandas_kwargs) if predict_csv is not None else None

        return cls.from_df(
            train_df,
            target,
            cat_cols,
            num_cols,
            valid_df,
            test_df,
            predict_df,
            batch_size,
            num_workers,
            val_size,
            test_size,
            preprocess_state=preprocess_state,
            preprocess_cls=preprocess_cls,
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
        valid_df: Optional[DataFrame] = None,
        test_df: Optional[DataFrame] = None,
        val_size: float = None,
        test_size: float = None,
    ):
        if valid_df is None and isinstance(val_size, float) and isinstance(test_size, float):
            assert 0 < val_size and val_size < 1
            assert 0 < test_size and test_size < 1
            train_df, valid_df = train_test_split(train_df, test_size=(val_size + test_size))

        if test_df is None and isinstance(test_size, float):
            assert 0 < test_size and test_size < 1
            valid_df, test_df = train_test_split(valid_df, test_size=test_size)

        return train_df, valid_df, test_df

    @staticmethod
    def _sanetize_cols(cat_cols: Optional[List], num_cols: Optional[List]):
        if cat_cols is None and num_cols is None:
            raise RuntimeError('Both `cat_cols` and `num_cols` are None!')

        cat_cols = cat_cols if cat_cols is not None else []
        num_cols = num_cols if num_cols is not None else []
        return cat_cols, num_cols

    @classmethod
    def from_df(
        cls,
        train_df: DataFrame,
        target: str,
        cat_cols: Optional[List] = None,
        num_cols: Optional[List] = None,
        valid_df: Optional[DataFrame] = None,
        test_df: Optional[DataFrame] = None,
        predict_df: Optional[DataFrame] = None,
        batch_size: int = 8,
        num_workers: Optional[int] = None,
        val_size: float = None,
        test_size: float = None,
        regression: bool = False,
        preprocess_state: Optional[TabularState] = None,
        preprocess_cls: Optional[Type[Preprocess]] = None,
    ):
        """Creates a TabularData object from pandas DataFrames.

        Args:
            train_df: train data DataFrame
            target: The column containing the class id.
            cat_cols: The list of categorical columns.
            num_cols: The list of numerical columns.
            valid_df: validation data DataFrame
            test_df: test data DataFrame
            batch_size: the batchsize to use for parallel loading. Defaults to 64.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to None which equals the number of available CPU threads,
            or 0 for Darwin platform.
            val_size: float between 0 and 1 to create a validation dataset from train dataset
            test_size: float between 0 and 1 to create a test dataset from train validation

        Returns:
            TabularData: The constructed data module.

        Examples::

            text_data = TextClassificationData.from_files("train.csv", label_field="class", text_field="sentence")
        """
        cat_cols, num_cols = cls._sanetize_cols(cat_cols, num_cols)

        train_df, valid_df, test_df = cls._split_dataframe(train_df, valid_df, test_df, val_size, test_size)

        preprocess_cls = preprocess_cls or cls.preprocess_cls

        preprocess_state = preprocess_cls.generate_state(
            train_df,
            valid_df,
            test_df,
            predict_df,
            target,
            num_cols,
            cat_cols,
            regression,
            preprocess_state=preprocess_state
        )

        preprocess = preprocess_cls.from_state(preprocess_state)

        return cls.from_load_data_inputs(
            train_load_data_input=train_df,
            valid_load_data_input=valid_df,
            test_load_data_input=test_df,
            predict_load_data_input=predict_df,
            batch_size=batch_size,
            num_workers=num_workers,
            preprocess=preprocess
        )
