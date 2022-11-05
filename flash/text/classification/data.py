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
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

from pandas.core.frame import DataFrame

from flash.core.data.data_module import DataModule
from flash.core.data.io.input import Input
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.utilities.classification import TargetFormatter
from flash.core.data.utilities.paths import PATH_TYPE
from flash.core.integrations.labelstudio.input import _parse_labelstudio_arguments, LabelStudioTextClassificationInput
from flash.core.utilities.imports import _TEXT_AVAILABLE, _TEXT_TESTING
from flash.core.utilities.stages import RunningStage
from flash.text.classification.input import (
    TextClassificationCSVInput,
    TextClassificationDataFrameInput,
    TextClassificationInput,
    TextClassificationJSONInput,
    TextClassificationListInput,
    TextClassificationParquetInput,
)

if _TEXT_AVAILABLE:
    from datasets import Dataset
else:
    Dataset = object

# Skip doctests if requirements aren't available
if not _TEXT_TESTING:
    __doctest_skip__ = ["TextClassificationData", "TextClassificationData.*"]


class TextClassificationData(DataModule):
    """The ``TextClassificationData`` class is a :class:`~flash.core.data.data_module.DataModule` with a set of
    classmethods for loading data for text classification."""

    input_transform_cls = InputTransform

    @classmethod
    def from_csv(
        cls,
        input_field: str,
        target_fields: Optional[Union[str, Sequence[str]]] = None,
        train_file: Optional[PATH_TYPE] = None,
        val_file: Optional[PATH_TYPE] = None,
        test_file: Optional[PATH_TYPE] = None,
        predict_file: Optional[PATH_TYPE] = None,
        target_formatter: Optional[TargetFormatter] = None,
        input_cls: Type[Input] = TextClassificationCSVInput,
        transform: Optional[Dict[str, Callable]] = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "TextClassificationData":
        """Load the :class:`~flash.text.classification.data.TextClassificationData` from CSV files containing text
        snippets and their corresponding targets.

        Input text snippets will be extracted from the ``input_field`` column in the CSV files.
        The targets will be extracted from the ``target_fields`` in the CSV files and can be in any of our
        :ref:`supported classification target formats <formatting_classification_targets>`.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            input_field: The field (column name) in the CSV files containing the text snippets.
            target_fields: The field (column name) or list of fields in the CSV files containing the targets.
            train_file: The CSV file to use when training.
            val_file: The CSV file to use when validating.
            test_file: The CSV file to use when testing.
            predict_file: The CSV file to use when predicting.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.text.classification.data.TextClassificationData`.

        Examples
        ________

        The files can be in Comma Separated Values (CSV) format with either a ``.csv`` or ``.txt`` extension.

        .. testsetup::

            >>> import os
            >>> from pandas import DataFrame
            >>> DataFrame.from_dict({
            ...     "reviews": ["Best movie ever!", "Not good", "Fine I guess"],
            ...     "targets": ["positive", "negative", "neutral"],
            ... }).to_csv("train_data.csv", index=False)
            >>> DataFrame.from_dict({
            ...     "reviews": ["Worst movie ever!", "I didn't enjoy it", "It was ok"],
            ... }).to_csv("predict_data.csv", index=False)

        The file ``train_data.csv`` contains the following:

        .. code-block::

            reviews,targets
            Best movie ever!,positive
            Not good,negative
            Fine I guess,neutral

        The file ``predict_data.csv`` contains the following:

        .. code-block::

            reviews
            Worst movie ever!
            I didn't enjoy it
            It was ok

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.text import TextClassifier, TextClassificationData
            >>> datamodule = TextClassificationData.from_csv(
            ...     "reviews",
            ...     "targets",
            ...     train_file="train_data.csv",
            ...     predict_file="predict_data.csv",
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            3
            >>> datamodule.labels
            ['negative', 'neutral', 'positive']
            >>> model = TextClassifier(num_classes=datamodule.num_classes, backbone="prajjwal1/bert-tiny")
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> os.remove("train_data.csv")
            >>> os.remove("predict_data.csv")

        Alternatively, the files can be in Tab Separated Values (TSV) format with a ``.tsv`` extension.

        .. testsetup::

            >>> import os
            >>> from pandas import DataFrame
            >>> DataFrame.from_dict({
            ...     "reviews": ["Best movie ever!", "Not good", "Fine I guess"],
            ...     "targets": ["positive", "negative", "neutral"],
            ... }).to_csv("train_data.tsv", sep="\\t", index=False)
            >>> DataFrame.from_dict({
            ...     "reviews": ["Worst movie ever!", "I didn't enjoy it", "It was ok"],
            ... }).to_csv("predict_data.tsv", sep="\\t", index=False)

        The file ``train_data.tsv`` contains the following:

        .. code-block::

            reviews             targets
            Best movie ever!    positive
            Not good            negative
            Fine I guess        neutral

        The file ``predict_data.tsv`` contains the following:

        .. code-block::

            reviews
            Worst movie ever!
            I didn't enjoy it
            It was ok

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.text import TextClassifier, TextClassificationData
            >>> datamodule = TextClassificationData.from_csv(
            ...     "reviews",
            ...     "targets",
            ...     train_file="train_data.tsv",
            ...     predict_file="predict_data.tsv",
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            3
            >>> datamodule.labels
            ['negative', 'neutral', 'positive']
            >>> model = TextClassifier(num_classes=datamodule.num_classes, backbone="prajjwal1/bert-tiny")
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> os.remove("train_data.tsv")
            >>> os.remove("predict_data.tsv")
        """
        ds_kw = dict(
            target_formatter=target_formatter,
            input_key=input_field,
            target_keys=target_fields,
        )

        train_input = input_cls(RunningStage.TRAINING, train_file, **ds_kw)
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
    def from_json(
        cls,
        input_field: str,
        target_fields: Optional[Union[str, Sequence[str]]] = None,
        train_file: Optional[PATH_TYPE] = None,
        val_file: Optional[PATH_TYPE] = None,
        test_file: Optional[PATH_TYPE] = None,
        predict_file: Optional[PATH_TYPE] = None,
        target_formatter: Optional[TargetFormatter] = None,
        input_cls: Type[Input] = TextClassificationJSONInput,
        transform: Optional[Dict[str, Callable]] = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        field: Optional[str] = None,
        **data_module_kwargs: Any,
    ) -> "TextClassificationData":
        """Load the :class:`~flash.text.classification.data.TextClassificationData` from JSON files containing text
        snippets and their corresponding targets.

        Input text snippets will be extracted from the ``input_field`` in the JSON objects.
        The targets will be extracted from the ``target_fields`` in the JSON objects and can be in any of our
        :ref:`supported classification target formats <formatting_classification_targets>`.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            input_field: The field in the JSON objects containing the text snippets.
            target_fields: The field or list of fields in the JSON objects containing the targets.
            train_file: The JSON file to use when training.
            val_file: The JSON file to use when validating.
            test_file: The JSON file to use when testing.
            predict_file: The JSON file to use when predicting.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            field: To specify the field that holds the data in the JSON file.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.text.classification.data.TextClassificationData`.

        Examples
        ________

        .. testsetup::

            >>> import os
            >>> from pandas import DataFrame
            >>> DataFrame.from_dict({
            ...     "reviews": ["Best movie ever!", "Not good", "Fine I guess"],
            ...     "targets": ["positive", "negative", "neutral"],
            ... }).to_json("train_data.json", orient="records", lines=True)
            >>> DataFrame.from_dict({
            ...     "reviews": ["Worst movie ever!", "I didn't enjoy it", "It was ok"],
            ... }).to_json("predict_data.json", orient="records", lines=True)

        The file ``train_data.json`` contains the following:

        .. code-block::

            {"reviews":"Best movie ever!","targets":"positive"}
            {"reviews":"Not good","targets":"negative"}
            {"reviews":"Fine I guess","targets":"neutral"}

        The file ``predict_data.json`` contains the following:

        .. code-block::

            {"reviews":"Worst movie ever!"}
            {"reviews":"I didn't enjoy it"}
            {"reviews":"It was ok"}

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.text import TextClassifier, TextClassificationData
            >>> datamodule = TextClassificationData.from_json(
            ...     "reviews",
            ...     "targets",
            ...     train_file="train_data.json",
            ...     predict_file="predict_data.json",
            ...     batch_size=2,
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Downloading...
            >>> datamodule.num_classes
            3
            >>> datamodule.labels
            ['negative', 'neutral', 'positive']
            >>> model = TextClassifier(num_classes=datamodule.num_classes, backbone="prajjwal1/bert-tiny")
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> os.remove("train_data.json")
            >>> os.remove("predict_data.json")
        """
        ds_kw = dict(
            target_formatter=target_formatter,
            input_key=input_field,
            target_keys=target_fields,
            field=field,
        )

        train_input = input_cls(RunningStage.TRAINING, train_file, **ds_kw)
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
    def from_parquet(
        cls,
        input_field: str,
        target_fields: Optional[Union[str, Sequence[str]]] = None,
        train_file: Optional[PATH_TYPE] = None,
        val_file: Optional[PATH_TYPE] = None,
        test_file: Optional[PATH_TYPE] = None,
        predict_file: Optional[PATH_TYPE] = None,
        target_formatter: Optional[TargetFormatter] = None,
        input_cls: Type[Input] = TextClassificationParquetInput,
        transform: Optional[Dict[str, Callable]] = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "TextClassificationData":
        """Load the :class:`~flash.text.classification.data.TextClassificationData` from PARQUET files containing
        text snippets and their corresponding targets.

        Input text snippets will be extracted from the ``input_field`` column in the PARQUET files.
        The targets will be extracted from the ``target_fields`` in the PARQUET files and can be in any of our
        :ref:`supported classification target formats <formatting_classification_targets>`.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            input_field: The field (column name) in the PARQUET files containing the text snippets.
            target_fields: The field (column name) or list of fields in the PARQUET files containing the targets.
            train_file: The PARQUET file to use when training.
            val_file: The PARQUET file to use when validating.
            test_file: The PARQUET file to use when testing.
            predict_file: The PARQUET file to use when predicting.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.text.classification.data.TextClassificationData`.

        Examples
        ________

        .. testsetup::

            >>> import os
            >>> from pandas import DataFrame
            >>> DataFrame.from_dict({
            ...     "reviews": ["Best movie ever!", "Not good", "Fine I guess"],
            ...     "targets": ["positive", "negative", "neutral"],
            ... }).to_parquet("train_data.parquet", index=False)
            >>> DataFrame.from_dict({
            ...     "reviews": ["Worst movie ever!", "I didn't enjoy it", "It was ok"],
            ... }).to_parquet("predict_data.parquet", index=False)

        The file ``train_data.parquet`` contains the following contents encoded in the PARQUET format:

        .. code-block::

            reviews,targets
            Best movie ever!,positive
            Not good,negative
            Fine I guess,neutral

        The file ``predict_data.parquet`` contains the following contents encoded in the PARQUET format:

        .. code-block::

            reviews
            Worst movie ever!
            I didn't enjoy it
            It was ok

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.text import TextClassifier, TextClassificationData
            >>> datamodule = TextClassificationData.from_parquet(
            ...     "reviews",
            ...     "targets",
            ...     train_file="train_data.parquet",
            ...     predict_file="predict_data.parquet",
            ...     batch_size=2,
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Downloading...
            >>> datamodule.num_classes
            3
            >>> datamodule.labels
            ['negative', 'neutral', 'positive']
            >>> model = TextClassifier(num_classes=datamodule.num_classes, backbone="prajjwal1/bert-tiny")
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> os.remove("train_data.parquet")
            >>> os.remove("predict_data.parquet")
        """
        ds_kw = dict(
            target_formatter=target_formatter,
            input_key=input_field,
            target_keys=target_fields,
        )

        train_input = input_cls(RunningStage.TRAINING, train_file, **ds_kw)
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
    def from_hf_datasets(
        cls,
        input_field: str,
        target_fields: Optional[Union[str, Sequence[str]]] = None,
        train_hf_dataset: Optional[Dataset] = None,
        val_hf_dataset: Optional[Dataset] = None,
        test_hf_dataset: Optional[Dataset] = None,
        predict_hf_dataset: Optional[Dataset] = None,
        target_formatter: Optional[TargetFormatter] = None,
        input_cls: Type[Input] = TextClassificationInput,
        transform: Optional[Dict[str, Callable]] = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "TextClassificationData":
        """Load the :class:`~flash.text.classification.data.TextClassificationData` from Hugging Face ``Dataset``
        objects containing text snippets and their corresponding targets.

        Input text snippets will be extracted from the ``input_field`` column in the ``Dataset`` objects.
        The targets will be extracted from the ``target_fields`` in the ``Dataset`` objects and can be in any of our
        :ref:`supported classification target formats <formatting_classification_targets>`.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            input_field: The field (column name) in the ``Dataset`` objects containing the text snippets.
            target_fields: The field (column name) or list of fields in the ``Dataset`` objects containing the targets.
            train_hf_dataset: The ``Dataset`` to use when training.
            val_hf_dataset: The ``Dataset`` to use when validating.
            test_hf_dataset: The ``Dataset`` to use when testing.
            predict_hf_dataset: The ``Dataset`` to use when predicting.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.text.classification.data.TextClassificationData`.

        Examples
        ________

        .. doctest::

            >>> from datasets import Dataset
            >>> from flash import Trainer
            >>> from flash.text import TextClassifier, TextClassificationData
            >>> train_data = Dataset.from_dict(
            ...     {
            ...         "reviews": ["Best movie ever!", "Not good", "Fine I guess"],
            ...         "targets": ["positive", "negative", "neutral"],
            ...     }
            ... )
            >>> predict_data = Dataset.from_dict(
            ...     {
            ...         "reviews": ["Worst movie ever!", "I didn't enjoy it", "It was ok"],
            ...     }
            ... )
            >>> datamodule = TextClassificationData.from_hf_datasets(
            ...     "reviews",
            ...     "targets",
            ...     train_hf_dataset=train_data,
            ...     predict_hf_dataset=predict_data,
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            3
            >>> datamodule.labels
            ['negative', 'neutral', 'positive']
            >>> model = TextClassifier(num_classes=datamodule.num_classes, backbone="prajjwal1/bert-tiny")
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
            input_key=input_field,
            target_keys=target_fields,
        )

        train_input = input_cls(RunningStage.TRAINING, train_hf_dataset, **ds_kw)
        ds_kw["target_formatter"] = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_hf_dataset, **ds_kw),
            input_cls(RunningStage.TESTING, test_hf_dataset, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_hf_dataset, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_data_frame(
        cls,
        input_field: str,
        target_fields: Optional[Union[str, Sequence[str]]] = None,
        train_data_frame: Optional[DataFrame] = None,
        val_data_frame: Optional[DataFrame] = None,
        test_data_frame: Optional[DataFrame] = None,
        predict_data_frame: Optional[DataFrame] = None,
        target_formatter: Optional[TargetFormatter] = None,
        input_cls: Type[Input] = TextClassificationDataFrameInput,
        transform: Optional[Dict[str, Callable]] = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "TextClassificationData":
        """Load the :class:`~flash.text.classification.data.TextClassificationData` from Pandas ``DataFrame``
        objects containing text snippets and their corresponding targets.

        Input text snippets will be extracted from the ``input_field`` column in the ``DataFrame`` objects.
        The targets will be extracted from the ``target_fields`` in the ``DataFrame`` objects and can be in any of our
        :ref:`supported classification target formats <formatting_classification_targets>`.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            input_field: The field (column name) in the ``DataFrame`` objects containing the text snippets.
            target_fields: The field (column name) or list of fields in the ``DataFrame`` objects containing the
                targets.
            train_data_frame: The ``DataFrame`` to use when training.
            val_data_frame: The ``DataFrame`` to use when validating.
            test_data_frame: The ``DataFrame`` to use when testing.
            predict_data_frame: The ``DataFrame`` to use when predicting.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.text.classification.data.TextClassificationData`.

        Examples
        ________

        .. doctest::

            >>> from pandas import DataFrame
            >>> from flash import Trainer
            >>> from flash.text import TextClassifier, TextClassificationData
            >>> train_data = DataFrame.from_dict(
            ...     {
            ...         "reviews": ["Best movie ever!", "Not good", "Fine I guess"],
            ...         "targets": ["positive", "negative", "neutral"],
            ...     }
            ... )
            >>> predict_data = DataFrame.from_dict(
            ...     {
            ...         "reviews": ["Worst movie ever!", "I didn't enjoy it", "It was ok"],
            ...     }
            ... )
            >>> datamodule = TextClassificationData.from_data_frame(
            ...     "reviews",
            ...     "targets",
            ...     train_data_frame=train_data,
            ...     predict_data_frame=predict_data,
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            3
            >>> datamodule.labels
            ['negative', 'neutral', 'positive']
            >>> model = TextClassifier(num_classes=datamodule.num_classes, backbone="prajjwal1/bert-tiny")
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
            input_key=input_field,
            target_keys=target_fields,
        )

        train_input = input_cls(RunningStage.TRAINING, train_data_frame, **ds_kw)
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
    def from_lists(
        cls,
        train_data: Optional[List[str]] = None,
        train_targets: Optional[Union[List[Any], List[List[Any]]]] = None,
        val_data: Optional[List[str]] = None,
        val_targets: Optional[Union[List[Any], List[List[Any]]]] = None,
        test_data: Optional[List[str]] = None,
        test_targets: Optional[Union[List[Any], List[List[Any]]]] = None,
        predict_data: Optional[List[str]] = None,
        target_formatter: Optional[TargetFormatter] = None,
        input_cls: Type[Input] = TextClassificationListInput,
        transform: Optional[Dict[str, Callable]] = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "TextClassificationData":
        """Load the :class:`~flash.text.classification.data.TextClassificationData` from lists of text snippets and
        corresponding lists of targets.

        The targets can be in any of our
        :ref:`supported classification target formats <formatting_classification_targets>`.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_data: The list of text snippets to use when training.
            train_targets: The list of targets to use when training.
            val_data: The list of text snippets to use when validating.
            val_targets: The list of targets to use when validating.
            test_data: The list of text snippets to use when testing.
            test_targets: The list of targets to use when testing.
            predict_data: The list of text snippets to use when predicting.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.text.classification.data.TextClassificationData`.

        Examples
        ________

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.text import TextClassifier, TextClassificationData
            >>> datamodule = TextClassificationData.from_lists(
            ...     train_data=["Best movie ever!", "Not good", "Fine I guess"],
            ...     train_targets=["positive", "negative", "neutral"],
            ...     predict_data=["Worst movie ever!", "I didn't enjoy it", "It was ok"],
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            3
            >>> datamodule.labels
            ['negative', 'neutral', 'positive']
            >>> model = TextClassifier(num_classes=datamodule.num_classes, backbone="prajjwal1/bert-tiny")
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...
        """
        ds_kw = dict(
            target_formatter=target_formatter,
        )

        train_input = input_cls(RunningStage.TRAINING, train_data, train_targets, **ds_kw)
        ds_kw["target_formatter"] = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_data, val_targets, **ds_kw),
            input_cls(RunningStage.TESTING, test_data, test_targets, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_data, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_labelstudio(
        cls,
        export_json: str = None,
        train_export_json: str = None,
        val_export_json: str = None,
        test_export_json: str = None,
        predict_export_json: str = None,
        data_folder: str = None,
        train_data_folder: str = None,
        val_data_folder: str = None,
        test_data_folder: str = None,
        predict_data_folder: str = None,
        input_cls: Type[Input] = LabelStudioTextClassificationInput,
        transform: Optional[Dict[str, Callable]] = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        val_split: Optional[float] = None,
        multi_label: Optional[bool] = False,
        **data_module_kwargs: Any,
    ) -> "TextClassificationData":
        """Creates a :class:`~flash.core.data.data_module.DataModule` object
        from the given export file and data directory using the
        :class:`~flash.core.data.io.input.Input` of name
        :attr:`~flash.core.data.io.input.InputFormat.FOLDERS`
        from the passed or constructed :class:`~flash.core.data.io.input_transform.InputTransform`.

        Args:
            export_json: path to label studio export file
            train_export_json: path to label studio export file for train set, overrides export_json if specified
            val_export_json: path to label studio export file for validation
            test_export_json: path to label studio export file for test
            predict_export_json: path to label studio export file for predict
            data_folder: path to label studio data folder
            train_data_folder: path to label studio data folder for train data set, overrides data_folder if specified
            val_data_folder: path to label studio data folder for validation data
            test_data_folder: path to label studio data folder for test data
            predict_data_folder: path to label studio data folder for predict data
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            multi_label: Whether the labels are multi encoded.
            data_module_kwargs: Additional keyword arguments to use when constructing the datamodule.

        Returns:
            The constructed data module.
        """

        train_data, val_data, test_data, predict_data = _parse_labelstudio_arguments(
            export_json=export_json,
            train_export_json=train_export_json,
            val_export_json=val_export_json,
            test_export_json=test_export_json,
            predict_export_json=predict_export_json,
            data_folder=data_folder,
            train_data_folder=train_data_folder,
            val_data_folder=val_data_folder,
            test_data_folder=test_data_folder,
            predict_data_folder=predict_data_folder,
            val_split=val_split,
            multi_label=multi_label,
        )

        ds_kw = dict()

        train_input = input_cls(RunningStage.TRAINING, train_data, **ds_kw)
        ds_kw["parameters"] = getattr(train_input, "parameters", None)

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_data, **ds_kw),
            input_cls(RunningStage.TESTING, test_data, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_data, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )
