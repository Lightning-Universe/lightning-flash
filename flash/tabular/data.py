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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from flash.core.classification import LabelsState
from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_module import DataModule
from flash.core.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources
from flash.core.data.process import Deserializer, Postprocess, Preprocess
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
        dfs = _pre_transform(
            [df], self.num_cols, self.cat_cols, self.codes, self.mean, self.std, self.target_col, self.target_codes
        )

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
        return [
            {DefaultDataKeys.INPUT: (c, n), DefaultDataKeys.TARGET: t} for c, n, t in zip(cat_vars, num_vars, target)
        ]

    def predict_load_data(self, data: DataFrame, dataset: Optional[Any] = None):
        _, cat_vars, num_vars = self.common_load_data(data, dataset=dataset)
        return [{DefaultDataKeys.INPUT: (c, n)} for c, n in zip(cat_vars, num_vars)]


class TabularCSVDataSource(TabularDataFrameDataSource):
    def load_data(self, data: str, dataset: Optional[Any] = None):
        return super().load_data(pd.read_csv(data), dataset=dataset)

    def predict_load_data(self, data: str, dataset: Optional[Any] = None):
        return super().predict_load_data(pd.read_csv(data), dataset=dataset)


class TabularDeserializer(Deserializer):
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
        self.classes = classes
        self.is_regression = is_regression

    def deserialize(self, data: str) -> Any:
        df = pd.read_csv(StringIO(data))
        df = _pre_transform(
            [df], self.num_cols, self.cat_cols, self.codes, self.mean, self.std, self.target_col, self.target_codes
        )[0]

        cat_vars = _to_cat_vars_numpy(df, self.cat_cols)
        num_vars = _to_num_vars_numpy(df, self.num_cols)

        cat_vars = np.stack(cat_vars, 1)
        num_vars = np.stack(num_vars, 1)

        return [{DefaultDataKeys.INPUT: [c, n]} for c, n in zip(cat_vars, num_vars)]

    @property
    def example_input(self) -> str:
        row = {}
        for cat_col in self.cat_cols:
            row[cat_col] = ["test"]
        for num_col in self.num_cols:
            row[num_col] = [0]
        return str(DataFrame.from_dict(row).to_csv())


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
        deserializer: Optional[Deserializer] = None,
    ):
        classes = classes or []

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
                "data_frame": TabularDataFrameDataSource(
                    cat_cols, num_cols, target_col, mean, std, codes, target_codes, classes, is_regression
                ),
            },
            default_data_source=DefaultDataSources.CSV,
            deserializer=deserializer
            or TabularDeserializer(
                cat_cols=cat_cols,
                num_cols=num_cols,
                target_col=target_col,
                mean=mean,
                std=std,
                codes=codes,
                target_codes=target_codes,
                classes=classes,
                is_regression=is_regression,
            ),
        )

    def get_state_dict(self, strict: bool = False) -> Dict[str, Any]:
        return {
            **self.transforms,
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
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = True) -> "Preprocess":
        return cls(**state_dict)


class TabularPostprocess(Postprocess):
    def uncollate(self, batch: Any) -> Any:
        return batch


class TabularData(DataModule):
    """Data module for tabular tasks."""

    preprocess_cls = TabularPreprocess
    postprocess_cls = TabularPostprocess

    is_regression: bool = False

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
    def embedding_sizes(self) -> list:
        """Recommended embedding sizes."""

        # https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
        # The following "formula" provides a general rule of thumb about the number of embedding dimensions:
        # embedding_dimensions =  number_of_categories**0.25
        num_classes = [len(self.codes[cat]) for cat in self.cat_cols]
        emb_dims = [max(int(n ** 0.25), 16) for n in num_classes]
        return list(zip(num_classes, emb_dims))

    @staticmethod
    def _sanetize_cols(cat_cols: Optional[Union[str, List[str]]], num_cols: Optional[Union[str, List[str]]]):
        if cat_cols is None and num_cols is None:
            raise RuntimeError("Both `cat_cols` and `num_cols` are None!")

        return cat_cols or [], num_cols or []

    @classmethod
    def compute_state(
        cls,
        train_data_frame: DataFrame,
        val_data_frame: Optional[DataFrame],
        test_data_frame: Optional[DataFrame],
        predict_data_frame: Optional[DataFrame],
        target_fields: str,
        numerical_fields: List[str],
        categorical_fields: List[str],
    ) -> Tuple[float, float, List[str], Dict[str, Any], Dict[str, Any]]:

        if train_data_frame is None:
            raise MisconfigurationException(
                "train_data_frame is required to instantiate the TabularDataFrameDataSource"
            )

        data_frames = [train_data_frame]

        if val_data_frame is not None:
            data_frames += [val_data_frame]

        if test_data_frame is not None:
            data_frames += [test_data_frame]

        if predict_data_frame is not None:
            data_frames += [predict_data_frame]

        mean, std = _compute_normalization(data_frames[0], numerical_fields)

        classes = list(data_frames[0][target_fields].unique())

        if data_frames[0][target_fields].dtype == object:
            # if the target_fields is a category, not an int
            target_codes = _generate_codes(data_frames, [target_fields])
        else:
            target_codes = None
        codes = _generate_codes(data_frames, categorical_fields)

        return mean, std, classes, codes, target_codes

    @classmethod
    def from_data_frame(
        cls,
        categorical_fields: Optional[Union[str, List[str]]],
        numerical_fields: Optional[Union[str, List[str]]],
        target_fields: Optional[str] = None,
        train_data_frame: Optional[DataFrame] = None,
        val_data_frame: Optional[DataFrame] = None,
        test_data_frame: Optional[DataFrame] = None,
        predict_data_frame: Optional[DataFrame] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        **preprocess_kwargs: Any,
    ):
        """Creates a :class:`~flash.tabular.data.TabularData` object from the given data frames.

        Args:
            categorical_fields: The field or fields (columns) in the CSV file containing categorical inputs.
            numerical_fields: The field or fields (columns) in the CSV file containing numerical inputs.
            target_fields: The field or fields (columns) in the CSV file to use for the target.
            train_data_frame: The pandas ``DataFrame`` containing the training data.
            val_data_frame: The pandas ``DataFrame`` containing the validation data.
            test_data_frame: The pandas ``DataFrame`` containing the testing data.
            predict_data_frame: The pandas ``DataFrame`` containing the data to use when predicting.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            data_fetcher: The :class:`~flash.core.data.callback.BaseDataFetcher` to pass to the
                :class:`~flash.core.data.data_module.DataModule`.
            preprocess: The :class:`~flash.core.data.data.Preprocess` to pass to the
                :class:`~flash.core.data.data_module.DataModule`. If ``None``, ``cls.preprocess_cls``
                will be constructed and used.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            batch_size: The ``batch_size`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            num_workers: The ``num_workers`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            preprocess_kwargs: Additional keyword arguments to use when constructing the preprocess. Will only be used
                if ``preprocess = None``.

        Returns:
            The constructed data module.

        Examples::

            data_module = TabularData.from_data_frame(
                "categorical_input",
                "numerical_input",
                "target",
                train_data_frame=train_data,
            )
        """
        categorical_fields, numerical_fields = cls._sanetize_cols(categorical_fields, numerical_fields)

        if not isinstance(categorical_fields, list):
            categorical_fields = [categorical_fields]

        if not isinstance(numerical_fields, list):
            numerical_fields = [numerical_fields]

        mean, std, classes, codes, target_codes = cls.compute_state(
            train_data_frame=train_data_frame,
            val_data_frame=val_data_frame,
            test_data_frame=test_data_frame,
            predict_data_frame=predict_data_frame,
            target_fields=target_fields,
            numerical_fields=numerical_fields,
            categorical_fields=categorical_fields,
        )

        return cls.from_data_source(
            "data_frame",
            train_data_frame,
            val_data_frame,
            test_data_frame,
            predict_data_frame,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            cat_cols=categorical_fields,
            num_cols=numerical_fields,
            target_col=target_fields,
            mean=mean,
            std=std,
            codes=codes,
            target_codes=target_codes,
            classes=classes,
            is_regression=cls.is_regression,
            **preprocess_kwargs,
        )

    @classmethod
    def from_csv(
        cls,
        categorical_fields: Optional[Union[str, List[str]]],
        numerical_fields: Optional[Union[str, List[str]]],
        target_fields: Optional[str] = None,
        train_file: Optional[str] = None,
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        predict_file: Optional[str] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        **preprocess_kwargs: Any,
    ) -> "DataModule":
        """Creates a :class:`~flash.tabular.data.TabularData` object from the given CSV files.

        Args:
            categorical_fields: The field or fields (columns) in the CSV file containing categorical inputs.
            numerical_fields: The field or fields (columns) in the CSV file containing numerical inputs.
            target_fields: The field or fields (columns) in the CSV file to use for the target.
            train_file: The CSV file containing the training data.
            val_file: The CSV file containing the validation data.
            test_file: The CSV file containing the testing data.
            predict_file: The CSV file containing the data to use when predicting.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            data_fetcher: The :class:`~flash.core.data.callback.BaseDataFetcher` to pass to the
                :class:`~flash.core.data.data_module.DataModule`.
            preprocess: The :class:`~flash.core.data.data.Preprocess` to pass to the
                :class:`~flash.core.data.data_module.DataModule`. If ``None``, ``cls.preprocess_cls``
                will be constructed and used.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            batch_size: The ``batch_size`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            num_workers: The ``num_workers`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            preprocess_kwargs: Additional keyword arguments to use when constructing the preprocess. Will only be used
                if ``preprocess = None``.

        Returns:
            The constructed data module.

        Examples::

            data_module = TabularData.from_csv(
                "categorical_input",
                "numerical_input",
                "target",
                train_file="train_data.csv",
            )
        """
        return cls.from_data_frame(
            categorical_fields=categorical_fields,
            numerical_fields=numerical_fields,
            target_fields=target_fields,
            train_data_frame=pd.read_csv(train_file) if train_file is not None else None,
            val_data_frame=pd.read_csv(val_file) if val_file is not None else None,
            test_data_frame=pd.read_csv(test_file) if test_file is not None else None,
            predict_data_frame=pd.read_csv(predict_file) if predict_file is not None else None,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
        )
