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
from typing import Optional, Union, List, Dict, Callable, Any

from flash.core.data.data_module import DataModule

from flash.core.data.process import Preprocess

from flash.core.data.callback import BaseDataFetcher

from flash.core.utilities.imports import _PANDAS_AVAILABLE

if _PANDAS_AVAILABLE:
    import pandas as pd
    from pandas.core.frame import DataFrame
else:
    DataFrame = object

from flash.tabular.data import TabularData


class TabularClassificationData(TabularData):
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
        """Creates a :class:`~flash.tabular.classification.data.TabularClassificationData` object from the given data
        frames.

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
            is_regression=False,
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
    ) -> 'DataModule':
        """Creates a :class:`~flash.tabular.classification.data.TabularClassificationData` object from the given CSV
        files.

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
            is_regression=False,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
        )
