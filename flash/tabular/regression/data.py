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
from typing import Any, Dict, List, Optional, Type, Union

from flash.core.data.data_pipeline import DataPipelineState
from flash.core.data.io.input import Input
from flash.core.data.io.input_transform import INPUT_TRANSFORM_TYPE, InputTransform
from flash.core.utilities.imports import _PANDAS_AVAILABLE, _TABULAR_AVAILABLE
from flash.core.utilities.stages import RunningStage
from flash.tabular.data import TabularData
from flash.tabular.regression.input import TabularRegressionCSVInput, TabularRegressionDataFrameInput

if _PANDAS_AVAILABLE:
    from pandas.core.frame import DataFrame
else:
    DataFrame = object

# Skip doctests if requirements aren't available
if not _TABULAR_AVAILABLE:
    __doctest_skip__ = ["TabularRegressionData", "TabularRegressionData.*"]


class TabularRegressionData(TabularData):
    """The ``TabularRegressionData`` class is a :class:`~flash.core.data.data_module.DataModule` with a set of
    classmethods for loading data for tabular regression."""

    @classmethod
    def from_data_frame(
        cls,
        categorical_fields: Optional[Union[str, List[str]]] = None,
        numerical_fields: Optional[Union[str, List[str]]] = None,
        target_field: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        train_data_frame: Optional[DataFrame] = None,
        val_data_frame: Optional[DataFrame] = None,
        test_data_frame: Optional[DataFrame] = None,
        predict_data_frame: Optional[DataFrame] = None,
        train_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        input_cls: Type[Input] = TabularRegressionDataFrameInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "TabularRegressionData":
        """Creates a :class:`~flash.tabular.regression.data.TabularRegressionData` object from the given data
        frames.

        .. note::

            The ``categorical_fields``, ``numerical_fields``, and ``target_field`` do not need to be provided if
            ``parameters`` are passed instead. These can be obtained from the
            :attr:`~flash.tabular.data.TabularData.parameters` attribute of the
            :class:`~flash.tabular.data.TabularData` object that contains your training data.

        The targets will be extracted from the ``target_field`` in the data frames.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            categorical_fields: The fields (column names) in the data frames containing categorical data.
            numerical_fields: The fields (column names) in the data frames containing numerical data.
            target_field: The field (column name) in the data frames containing the targets.
            parameters: Parameters to use if ``categorical_fields``, ``numerical_fields``, and ``target_field`` are not
                provided (e.g. when loading data for inference or validation).
            train_data_frame: The pandas DataFrame to use when training.
            val_data_frame: The pandas DataFrame to use when validating.
            test_data_frame: The pandas DataFrame to use when testing.
            predict_data_frame: The pandas DataFrame to use when predicting.
            train_transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use when training.
            val_transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use when validating.
            test_transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use when testing.
            predict_transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use when
                predicting.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
              :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.tabular.regression.data.TabularRegressionData`.

        Examples
        ________

        .. testsetup::

            >>> from pandas import DataFrame
            >>> train_data = DataFrame.from_dict({
            ...     "age": [2, 4, 1],
            ...     "animal": ["cat", "dog", "cat"],
            ...     "weight": [6, 10, 5],
            ... })
            >>> predict_data = DataFrame.from_dict({
            ...     "animal": ["dog", "dog", "cat"],
            ...     "weight": [7, 12, 5],
            ... })

        We have a DataFrame ``train_data`` with the following contents:

        .. doctest::

            >>> train_data.head(3)
               age animal  weight
            0    2    cat       6
            1    4    dog      10
            2    1    cat       5

        and a DataFrame ``predict_data`` with the following contents:

        .. doctest::

            >>> predict_data.head(3)
              animal  weight
            0    dog       7
            1    dog      12
            2    cat       5

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.tabular import TabularRegressor, TabularRegressionData
            >>> datamodule = TabularRegressionData.from_data_frame(
            ...     "animal",
            ...     "weight",
            ...     "age",
            ...     train_data_frame=train_data,
            ...     predict_data_frame=predict_data,
            ...     batch_size=4,
            ... )
            >>> model = TabularRegressor.from_data(datamodule, backbone="tabnet")
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...
        """
        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
            categorical_fields=categorical_fields,
            numerical_fields=numerical_fields,
            target_field=target_field,
            parameters=parameters,
        )

        train_input = input_cls(RunningStage.TRAINING, train_data_frame, transform=train_transform, **ds_kw)

        ds_kw["parameters"] = train_input.parameters if train_input else parameters

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_data_frame, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_data_frame, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_data_frame, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )

    @classmethod
    def from_csv(
        cls,
        categorical_fields: Optional[Union[str, List[str]]] = None,
        numerical_fields: Optional[Union[str, List[str]]] = None,
        target_field: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        train_file: Optional[str] = None,
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        predict_file: Optional[str] = None,
        train_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        input_cls: Type[Input] = TabularRegressionCSVInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "TabularRegressionData":
        """Creates a :class:`~flash.tabular.regression.data.TabularRegressionData` object from the given CSV files.

        .. note::

            The ``categorical_fields``, ``numerical_fields``, and ``target_field`` do not need to be provided if
            ``parameters`` are passed instead. These can be obtained from the
            :attr:`~flash.tabular.data.TabularData.parameters` attribute of the
            :class:`~flash.tabular.data.TabularData` object that contains your training data.

        The targets will be extracted from the ``target_field`` in the CSV files.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            categorical_fields: The fields (column names) in the CSV files containing categorical data.
            numerical_fields: The fields (column names) in the CSV files containing numerical data.
            target_field: The field (column name) in the CSV files containing the targets.
            parameters: Parameters to use if ``categorical_fields``, ``numerical_fields``, and ``target_field`` are not
                provided (e.g. when loading data for inference or validation).
            train_file: The CSV file to use when training.
            val_file: The CSV file to use when validating.
            test_file: The CSV file to use when testing.
            predict_file: The CSV file to use when predicting.
            train_transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use when training.
            val_transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use when validating.
            test_transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use when testing.
            predict_transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use when
                predicting.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
              :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.tabular.regression.data.TabularRegressionData`.

        Examples
        ________

        .. testsetup::

            >>> from pandas import DataFrame
            >>> DataFrame.from_dict({
            ...     "age": [2, 4, 1],
            ...     "animal": ["cat", "dog", "cat"],
            ...     "weight": [6, 10, 5],
            ... }).to_csv("train_data.csv")
            >>> DataFrame.from_dict({
            ...     "animal": ["dog", "dog", "cat"],
            ...     "weight": [7, 12, 5],
            ... }).to_csv("predict_data.csv")

        We have a ``train_data.csv`` with the following contents:

        .. code-block::

            age,animal,weight
            2,cat,6
            4,dog,10
            1,cat,5

        and a ``predict_data.csv`` with the following contents:

        .. code-block::

            animal,weight
            dog,7
            dog,12
            cat,5

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.tabular import TabularRegressor, TabularRegressionData
            >>> datamodule = TabularRegressionData.from_csv(
            ...     "animal",
            ...     "weight",
            ...     "age",
            ...     train_file="train_data.csv",
            ...     predict_file="predict_data.csv",
            ...     batch_size=4,
            ... )
            >>> model = TabularRegressor.from_data(datamodule, backbone="tabnet")
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> import os
            >>> os.remove("train_data.csv")
            >>> os.remove("predict_data.csv")
        """
        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
            categorical_fields=categorical_fields,
            numerical_fields=numerical_fields,
            target_field=target_field,
            parameters=parameters,
        )

        train_input = input_cls(RunningStage.TRAINING, train_file, transform=train_transform, **ds_kw)

        ds_kw["parameters"] = train_input.parameters if train_input else parameters

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_file, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_file, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_file, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )
