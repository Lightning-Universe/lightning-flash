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
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pytorch_lightning.trainer.states import RunningStage
from sklearn.model_selection import train_test_split

from flash.data.auto_dataset import AutoDataset
from flash.data.data_module import DataModule
from flash.data.data_pipeline import DataPipeline
from flash.data.process import Preprocess
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
class TabularState:
    mean: DataFrame
    std: DataFrame
    codes: Dict
    target_codes: Optional[Dict]
    num_classes: int


class TabularPreprocess(Preprocess):

    def __init__(
        self,
        categorical_input: List,
        numerical_input: List,
        target: str,
        mean: DataFrame = None,
        std: DataFrame = None,
        codes: Dict = None,
        target_codes: Dict = None,
        regression: bool = False,
    ):
        super().__init__()
        self.categorical_input = categorical_input
        self.numerical_input = numerical_input
        self.target = target
        self.mean = mean
        self.std = std
        self.codes = codes
        self.target_codes = target_codes
        self.regression = regression

    @staticmethod
    def _generate_state(dfs: List[DataFrame], target: str, numerical_input: List, categorical_input: List):
        mean, std = _compute_normalization(dfs[0], numerical_input)
        codes = _generate_codes(dfs, [target])
        num_classes = len(dfs[0][target].unique())
        if dfs[0][target].dtype == object:
            # if the target is a category, not an int
            target_codes = _generate_codes(dfs, [target])
        else:
            target_codes = None
        codes = _generate_codes(dfs, categorical_input)
        return TabularState(mean, std, codes, target_codes, num_classes)

    def common_load_data(self, df: DataFrame, dataset: AutoDataset):
        # impute_data
        dfs = _impute([df], self.numerical_input)

        # compute train dataset stats
        dfs = _pre_transform(
            dfs, self.numerical_input, self.categorical_input, self.codes, self.mean, self.std, self.target,
            self.target_codes
        )

        df = dfs[0]

        dataset.num_samples = len(df)
        cat_vars = _to_cat_vars_numpy(df, self.categorical_input)
        num_vars = _to_num_cols_numpy(df, self.numerical_input)
        dataset.num_samples = len(df)
        cat_vars = np.stack(cat_vars, 1) if len(cat_vars) else np.zeros((len(self), 0))
        num_vars = np.stack(num_vars, 1) if len(num_vars) else np.zeros((len(self), 0))
        return df, cat_vars, num_vars

    def load_data(self, df: DataFrame, dataset: AutoDataset):
        df, cat_vars, num_vars = self.common_load_data(df, dataset)
        target = df[self.target].to_numpy().astype(np.float32 if self.regression else np.int64)
        return [((c, n), t) for c, n, t in zip(cat_vars, num_vars, target)]

    def predict_load_data(self, df: DataFrame, dataset: AutoDataset):
        _, cat_vars, num_vars = self.common_load_data(df, dataset)
        return [((c, n), -1) for c, n in zip(cat_vars, num_vars)]


class TabularData(DataModule):
    """Data module for tabular tasks"""

    preprocess_cls = TabularPreprocess

    @property
    def preprocess_state(self):
        return self._preprocess_state

    @preprocess_state.setter
    def preprocess_state(self, preprocess_state):
        self._preprocess_state = preprocess_state

    def __init__(
        self,
        train_df: DataFrame,
        target: str,
        categorical_input: Optional[List] = None,
        numerical_input: Optional[List] = None,
        valid_df: Optional[DataFrame] = None,
        test_df: Optional[DataFrame] = None,
        predict_df: Optional[DataFrame] = None,
        batch_size: int = 2,
        num_workers: Optional[int] = None,
    ):
        if categorical_input is None and numerical_input is None:
            raise RuntimeError('Both `categorical_input` and `numerical_input` are None!')

        categorical_input = categorical_input if categorical_input is not None else []
        numerical_input = numerical_input if numerical_input is not None else []

        self.cat_cols = categorical_input
        self.num_cols = numerical_input
        self.target = target

        self._preprocess_state = None

        if isinstance(train_df, DataFrame):
            dfs = [train_df]
            if valid_df is not None:
                dfs += [valid_df]
            if test_df is not None:
                dfs += [test_df]
            if predict_df is not None:
                dfs += [predict_df]
            self._preprocess_state = self.preprocess_cls._generate_state(
                dfs, target, numerical_input, categorical_input
            )

        train_ds = self._generate_dataset_if_possible(
            train_df, running_stage=RunningStage.TRAINING, data_pipeline=self.data_pipeline
        )
        valid_ds = self._generate_dataset_if_possible(
            valid_df, running_stage=RunningStage.VALIDATING, data_pipeline=self.data_pipeline
        )
        test_ds = self._generate_dataset_if_possible(
            test_df, running_stage=RunningStage.TESTING, data_pipeline=self.data_pipeline
        )
        predict_ds = self._generate_dataset_if_possible(
            predict_df, running_stage=RunningStage.PREDICTING, data_pipeline=self.data_pipeline
        )

        super().__init__(
            train_ds=train_ds,
            valid_ds=valid_ds,
            test_ds=test_ds,
            predict_ds=predict_ds,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        """
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
        """

    @property
    def codes(self):
        return self._preprocess_state.codes

    @property
    def num_classes(self) -> int:
        return self._preprocess_state.num_classes

    @property
    def num_features(self) -> int:
        return len(self.cat_cols) + len(self.num_cols)

    @property
    def preprocess(self):
        mean = None
        std = None
        codes = None

        if isinstance(self._preprocess_state, TabularState):
            mean = self._preprocess_state.mean
            std = self._preprocess_state.std
            codes = self._preprocess_state.codes

        return self.preprocess_cls(
            categorical_input=self.cat_cols,
            numerical_input=self.num_cols,
            target=self.target,
            mean=mean,
            std=std,
            codes=codes,
        )

    @classmethod
    def _generate_dataset_if_possible(
        cls,
        data: Optional[Any],
        running_stage: RunningStage,
        whole_data_load_fn: Optional[Callable] = None,
        per_sample_load_fn: Optional[Callable] = None,
        data_pipeline: Optional[DataPipeline] = None
    ) -> Optional[AutoDataset]:
        if data is None:
            return None

        if data_pipeline is not None:
            return data_pipeline._generate_auto_dataset(data, running_stage=running_stage)

        return cls.autogenerate_dataset(data, running_stage, whole_data_load_fn, per_sample_load_fn, data_pipeline)

    @classmethod
    def from_csv(
        cls,
        target: str,
        train_csv: Optional[str] = None,
        categorical_input: Optional[List] = None,
        numerical_input: Optional[List] = None,
        valid_csv: Optional[str] = None,
        test_csv: Optional[str] = None,
        predict_csv: Optional[str] = None,
        batch_size: int = 8,
        num_workers: Optional[int] = None,
        val_size: Optional[float] = None,
        test_size: Optional[float] = None,
        data_pipeline: Optional[DataPipeline] = None,
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
        predict_df = pd.read_csv(predict_csv, **pandas_kwargs) if predict_csv is not None else None

        datamodule = cls.from_df(
            train_df, target, categorical_input, numerical_input, valid_df, test_df, predict_df, batch_size,
            num_workers, val_size, test_size
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

    @classmethod
    def from_df(
        cls,
        train_df: DataFrame,
        target: str,
        categorical_input: Optional[List] = None,
        numerical_input: Optional[List] = None,
        valid_df: Optional[DataFrame] = None,
        test_df: Optional[DataFrame] = None,
        predict_df: Optional[DataFrame] = None,
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

        return datamodule
