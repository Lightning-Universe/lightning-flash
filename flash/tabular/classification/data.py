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

from flash.core.data.io.input import Input
from flash.core.data.io.input_transform import INPUT_TRANSFORM_TYPE, InputTransform
from flash.core.data.utilities.classification import TargetFormatter
from flash.core.utilities.imports import _PANDAS_AVAILABLE, _TABULAR_TESTING
from flash.core.utilities.stages import RunningStage
from flash.tabular.classification.input import (
    TabularClassificationCSVInput,
    TabularClassificationDataFrameInput,
    TabularClassificationDictInput,
    TabularClassificationListInput,
)
from flash.tabular.data import TabularData

if _PANDAS_AVAILABLE:
    from pandas.core.frame import DataFrame
else:
    DataFrame = object

# Skip doctests if requirements aren't available
if not _TABULAR_TESTING:
    __doctest_skip__ = ["TabularClassificationData", "TabularClassificationData.*"]


class TabularClassificationData(TabularData):
    """The ``TabularClassificationData`` class is a :class:`~flash.core.data.data_module.DataModule` with a set of
    classmethods for loading data for tabular classification."""

    @classmethod
    def from_data_frame(
        cls,
        categorical_fields: Optional[Union[str, List[str]]] = None,
        numerical_fields: Optional[Union[str, List[str]]] = None,
        target_fields: Optional[Union[str, List[str]]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        train_data_frame: Optional[DataFrame] = None,
        val_data_frame: Optional[DataFrame] = None,
        test_data_frame: Optional[DataFrame] = None,
        predict_data_frame: Optional[DataFrame] = None,
        target_formatter: Optional[TargetFormatter] = None,
        input_cls: Type[Input] = TabularClassificationDataFrameInput,
        transform: INPUT_TRANSFORM_TYPE = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "TabularClassificationData":
        """Creates a :class:`~flash.tabular.classification.data.TabularClassificationData` object from the given
        data frames.

        .. note::

            The ``categorical_fields``, ``numerical_fields``, and ``target_fields`` do not need to be provided if
            ``parameters`` are passed instead. These can be obtained from the
            :attr:`~flash.tabular.data.TabularData.parameters` attribute of the
            :class:`~flash.tabular.data.TabularData` object that contains your training data.

        The targets will be extracted from the ``target_fields`` in the data frames and can be in any of our
        :ref:`supported classification target formats <formatting_classification_targets>`.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            categorical_fields: The fields (column names) in the data frames containing categorical data.
            numerical_fields: The fields (column names) in the data frames containing numerical data.
            target_fields: The field (column name) or list of fields in the data frames containing the targets.
            parameters: Parameters to use if ``categorical_fields``, ``numerical_fields``, and ``target_fields`` are not
                provided (e.g. when loading data for inference or validation).
            train_data_frame: The DataFrame to use when training.
            val_data_frame: The DataFrame to use when validating.
            test_data_frame: The DataFrame to use when testing.
            predict_data_frame: The DataFrame to use when predicting.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.tabular.classification.data.TabularClassificationData`.

        Examples
        ________

        .. testsetup::

            >>> from pandas import DataFrame
            >>> train_data = DataFrame.from_dict({
            ...     "animal": ["cat", "dog", "cat"],
            ...     "friendly": ["yes", "yes", "no"],
            ...     "weight": [6, 10, 5],
            ... })
            >>> predict_data = DataFrame.from_dict({
            ...     "friendly": ["yes", "no", "yes"],
            ...     "weight": [7, 12, 5],
            ... })

        We have a DataFrame ``train_data`` with the following contents:

        .. doctest::

            >>> train_data.head(3)
              animal friendly  weight
            0    cat      yes       6
            1    dog      yes      10
            2    cat       no       5

        and a DataFrame ``predict_data`` with the following contents:

        .. doctest::

            >>> predict_data.head(3)
              friendly  weight
            0      yes       7
            1       no      12
            2      yes       5

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.tabular import TabularClassifier, TabularClassificationData
            >>> datamodule = TabularClassificationData.from_data_frame(
            ...     "friendly",
            ...     "weight",
            ...     "animal",
            ...     train_data_frame=train_data,
            ...     predict_data_frame=predict_data,
            ...     batch_size=4,
            ... )
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['cat', 'dog']
            >>> model = TabularClassifier.from_data(datamodule, backbone="tabnet")
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> del train_data
            >>> del predict_data
        """
        ds_kw = dict(
            target_formatter=target_formatter,
            categorical_fields=categorical_fields,
            numerical_fields=numerical_fields,
            target_fields=target_fields,
            parameters=parameters,
        )

        train_input = input_cls(RunningStage.TRAINING, train_data_frame, **ds_kw)
        ds_kw["parameters"] = train_input.parameters if train_input else parameters
        ds_kw["target_formatter"] = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_data_frame, **ds_kw),
            input_cls(RunningStage.TESTING, test_data_frame, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_data_frame, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_csv(
        cls,
        categorical_fields: Optional[Union[str, List[str]]] = None,
        numerical_fields: Optional[Union[str, List[str]]] = None,
        target_fields: Optional[Union[str, List[str]]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        train_file: Optional[str] = None,
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        predict_file: Optional[str] = None,
        target_formatter: Optional[TargetFormatter] = None,
        input_cls: Type[Input] = TabularClassificationCSVInput,
        transform: INPUT_TRANSFORM_TYPE = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "TabularClassificationData":
        """Creates a :class:`~flash.tabular.classification.data.TabularClassificationData` object from the given
        CSV files.

        .. note::

            The ``categorical_fields``, ``numerical_fields``, and ``target_fields`` do not need to be provided if
            ``parameters`` are passed instead. These can be obtained from the
            :attr:`~flash.tabular.data.TabularData.parameters` attribute of the
            :class:`~flash.tabular.data.TabularData` object that contains your training data.

        The targets will be extracted from the ``target_fields`` columns in the CSV files and can be in any of our
        :ref:`supported classification target formats <formatting_classification_targets>`.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            categorical_fields: The fields (column names) in the CSV files containing categorical data.
            numerical_fields: The fields (column names) in the CSV files containing numerical data.
            target_fields: The field (column name) or list of fields in the CSV files containing the targets.
            parameters: Parameters to use if ``categorical_fields``, ``numerical_fields``, and ``target_fields`` are not
                provided (e.g. when loading data for inference or validation).
            train_file: The path to the CSV file to use when training.
            val_file: The path to the CSV file to use when validating.
            test_file: The path to the CSV file to use when testing.
            predict_file: The path to the CSV file to use when predicting.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
              :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.tabular.classification.data.TabularClassificationData`.

        Examples
        ________

        The files can be in Comma Separated Values (CSV) format with either a ``.csv`` or ``.txt`` extension.

        .. testsetup::

            >>> from pandas import DataFrame
            >>> DataFrame.from_dict({
            ...     "animal": ["cat", "dog", "cat"],
            ...     "friendly": ["yes", "yes", "no"],
            ...     "weight": [6, 10, 5],
            ... }).to_csv("train_data.csv")
            >>> predict_data = DataFrame.from_dict({
            ...     "friendly": ["yes", "no", "yes"],
            ...     "weight": [7, 12, 5],
            ... }).to_csv("predict_data.csv")

        We have a ``train_data.csv`` with the following contents:

        .. code-block::

            animal,friendly,weight
            cat,yes,6
            dog,yes,10
            cat,no,5

        and a ``predict_data.csv`` with the following contents:

        .. code-block::

            friendly,weight
            yes,7
            no,12
            yes,5

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.tabular import TabularClassifier, TabularClassificationData
            >>> datamodule = TabularClassificationData.from_csv(
            ...     "friendly",
            ...     "weight",
            ...     "animal",
            ...     train_file="train_data.csv",
            ...     predict_file="predict_data.csv",
            ...     batch_size=4,
            ... )
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['cat', 'dog']
            >>> model = TabularClassifier.from_data(datamodule, backbone="tabnet")
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> import os
            >>> os.remove("train_data.csv")
            >>> os.remove("predict_data.csv")

        Alternatively, the files can be in Tab Separated Values (TSV) format with a ``.tsv`` extension.

        .. testsetup::

            >>> from pandas import DataFrame
            >>> DataFrame.from_dict({
            ...     "animal": ["cat", "dog", "cat"],
            ...     "friendly": ["yes", "yes", "no"],
            ...     "weight": [6, 10, 5],
            ... }).to_csv("train_data.tsv", sep="\\t")
            >>> predict_data = DataFrame.from_dict({
            ...     "friendly": ["yes", "no", "yes"],
            ...     "weight": [7, 12, 5],
            ... }).to_csv("predict_data.tsv", sep="\\t")

        We have a ``train_data.tsv`` with the following contents:

        .. code-block::

            animal  friendly    weight
            cat     yes         6
            dog     yes         10
            cat     no          5

        and a ``predict_data.tsv`` with the following contents:

        .. code-block::

            friendly    weight
            yes         7
            no          12
            yes         5

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.tabular import TabularClassifier, TabularClassificationData
            >>> datamodule = TabularClassificationData.from_csv(
            ...     "friendly",
            ...     "weight",
            ...     "animal",
            ...     train_file="train_data.tsv",
            ...     predict_file="predict_data.tsv",
            ...     batch_size=4,
            ... )
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['cat', 'dog']
            >>> model = TabularClassifier.from_data(datamodule, backbone="tabnet")
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> import os
            >>> os.remove("train_data.tsv")
            >>> os.remove("predict_data.tsv")
        """
        ds_kw = dict(
            target_formatter=target_formatter,
            categorical_fields=categorical_fields,
            numerical_fields=numerical_fields,
            target_fields=target_fields,
            parameters=parameters,
        )

        train_input = input_cls(RunningStage.TRAINING, train_file, **ds_kw)
        ds_kw["parameters"] = train_input.parameters if train_input else parameters
        ds_kw["target_formatter"] = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_file, **ds_kw),
            input_cls(RunningStage.TESTING, test_file, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_file, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_dicts(
        cls,
        categorical_fields: Optional[Union[str, List[str]]] = None,
        numerical_fields: Optional[Union[str, List[str]]] = None,
        target_fields: Optional[Union[str, List[str]]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        train_dict: Optional[Dict[str, List[Any]]] = None,
        val_dict: Optional[Dict[str, List[Any]]] = None,
        test_dict: Optional[Dict[str, List[Any]]] = None,
        predict_dict: Optional[Dict[str, List[Any]]] = None,
        target_formatter: Optional[TargetFormatter] = None,
        input_cls: Type[Input] = TabularClassificationDictInput,
        transform: INPUT_TRANSFORM_TYPE = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "TabularClassificationData":
        """Creates a :class:`~flash.tabular.classification.data.TabularClassificationData` object from the given
        dictionary.

        .. note::
            The ``categorical_fields``, ``numerical_fields``, and ``target_fields`` do not need to be provided if
            ``parameters`` are passed instead. These can be obtained from the
            :attr:`~flash.tabular.data.TabularData.parameters` attribute of the
            :class:`~flash.tabular.data.TabularData` object that contains your training data.

        The targets will be extracted from the ``target_fields`` in the dict and can be in any of our
        :ref:`supported classification target formats <formatting_classification_targets>`.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            categorical_fields: The fields (column names) in the dictionary containing categorical data.
            numerical_fields: The fields (column names) in the dictionary containing numerical data.
            target_fields: The field (column name) or list of fields in the dictionary containing the targets.
            parameters: Parameters to use if ``categorical_fields``, ``numerical_fields``, and ``target_fields`` are not
                provided (e.g. when loading data for inference or validation).
            train_dict: The data to use when training.
            val_dict: The data to use when validating.
            test_dict: The data to use when testing.
            predict_dict: The data to use when predicting.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.tabular.classification.data.TabularClassificationData`.

        Examples
        ________

        .. testsetup::

            >>> train_data = {
            ...     "animal": ["cat", "dog", "cat"],
            ...     "friendly": ["yes", "yes", "no"],
            ...     "weight": [6, 10, 5],
            ... }
            >>> predict_data = {
            ...     "friendly": ["yes", "no", "yes"],
            ...     "weight": [7, 12, 5],
            ... }

        We have a dictionary ``train_data`` with the following contents:

        .. code-block::

            {
                "age": [2, 4, 1],
                "animal": ["cat", "dog", "cat"],
                "weight": [6, 10, 5]
            }

        and a dictionary ``predict_data`` with the following contents:

        .. code-block::

            {
                "animal": ["dog", "dog", "cat"],
                "weight": [7, 12, 5]
            }

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.tabular import TabularClassifier, TabularClassificationData
            >>> datamodule = TabularClassificationData.from_dicts(
            ...     "friendly",
            ...     "weight",
            ...     "animal",
            ...     train_dict=train_data,
            ...     predict_dict=predict_data,
            ...     batch_size=4,
            ... )
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['cat', 'dog']
            >>> model = TabularClassifier.from_data(datamodule, backbone="tabnet")
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> del train_data
            >>> del predict_data
        """
        ds_kw = dict(
            target_formatter=target_formatter,
            categorical_fields=categorical_fields,
            numerical_fields=numerical_fields,
            target_fields=target_fields,
            parameters=parameters,
        )

        train_input = input_cls(RunningStage.TRAINING, train_dict, **ds_kw)
        ds_kw["parameters"] = train_input.parameters if train_input else parameters
        ds_kw["target_formatter"] = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_dict, **ds_kw),
            input_cls(RunningStage.TESTING, test_dict, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_dict, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_lists(
        cls,
        categorical_fields: Optional[Union[str, List[str]]] = None,
        numerical_fields: Optional[Union[str, List[str]]] = None,
        target_fields: Optional[Union[str, List[str]]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        train_list: Optional[List[Union[tuple, dict]]] = None,
        val_list: Optional[List[Union[tuple, dict]]] = None,
        test_list: Optional[List[Union[tuple, dict]]] = None,
        predict_list: Optional[List[Union[tuple, dict]]] = None,
        target_formatter: Optional[TargetFormatter] = None,
        input_cls: Type[Input] = TabularClassificationListInput,
        transform: INPUT_TRANSFORM_TYPE = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "TabularClassificationData":
        """Creates a :class:`~flash.tabular.classification.data.TabularClassificationData` object from the given
        data (in the form of list of a tuple or a dictionary).

        .. note::
            The ``categorical_fields``, ``numerical_fields``, and ``target_fields`` do not need to be provided if
            ``parameters`` are passed instead. These can be obtained from the
            :attr:`~flash.tabular.data.TabularData.parameters` attribute of the
            :class:`~flash.tabular.data.TabularData` object that contains your training data.

        The targets will be extracted from the ``target_fields`` in the dict and can be in any of our
        :ref:`supported classification target formats <formatting_classification_targets>`.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            categorical_fields: The fields (column names) in the dictionary containing categorical data.
            numerical_fields: The fields (column names) in the dictionary containing numerical data.
            target_fields: The field (column name) or list of fields in the dictionary containing the targets.
            parameters: Parameters to use if ``categorical_fields``, ``numerical_fields``, and ``target_fields`` are not
                provided (e.g. when loading data for inference or validation).
            train_list: The data to use when training.
            val_list: The data to use when validating.
            test_lists The data to use when testing.
            predict_list: The data to use when predicting.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.tabular.classification.data.TabularClassificationData`.

        Examples
        ________

        .. testsetup::

            >>> train_data = [
            ...     {"animal": "cat", "friendly": "yes", "weight": 6},
            ...     {"animal": "dog", "friendly": "yes", "weight": 10},
            ...     {"animal": "cat", "friendly": "no", "weight": 5},
            ... ]
            >>> predict_data = [
            ...     {"friendly": "yes", "weight": 7},
            ...     {"friendly": "no", "weight": 12},
            ...     {"friendly": "yes", "weight": 5},
            ... ]

        We have a list of dictionaries ``train_data`` with the following contents:

        .. code-block::

            [
                {"animal": "cat", "friendly": "yes", "weight": 6},
                {"animal": "dog", "friendly": "yes", "weight": 10},
                {"animal": "cat", "friendly": "no", "weight": 5},
            ]

        and a list of dictionaries ``predict_data`` with the following contents:

        .. code-block::

            [
                {"friendly": "yes", "weight": 7},
                {"friendly": "no", "weight": 12},
                {"friendly": "yes", "weight": 5},
            ]

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.tabular import TabularClassifier, TabularClassificationData
            >>> datamodule = TabularClassificationData.from_lists(
            ...     "friendly",
            ...     "weight",
            ...     "animal",
            ...     train_list=train_data,
            ...     predict_list=predict_data,
            ...     batch_size=4,
            ... )
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['cat', 'dog']
            >>> model = TabularClassifier.from_data(datamodule, backbone="tabnet")
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> del train_data
            >>> del predict_data
        """
        ds_kw = dict(
            target_formatter=target_formatter,
            categorical_fields=categorical_fields,
            numerical_fields=numerical_fields,
            target_fields=target_fields,
            parameters=parameters,
        )

        train_input = input_cls(RunningStage.TRAINING, train_list, **ds_kw)
        ds_kw["parameters"] = train_input.parameters if train_input else parameters
        ds_kw["target_formatter"] = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_list, **ds_kw),
            input_cls(RunningStage.TESTING, test_list, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_list, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )
