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
from typing import Any, Dict, List, Optional, Type

from flash.core.data.data_module import DataModule
from flash.core.data.io.input import Input
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.utilities.paths import PATH_TYPE
from flash.core.utilities.imports import _TEXT_AVAILABLE, _TEXT_TESTING
from flash.core.utilities.stages import RunningStage
from flash.core.utilities.types import INPUT_TRANSFORM_TYPE
from flash.text.seq2seq.core.input import Seq2SeqCSVInput, Seq2SeqInputBase, Seq2SeqJSONInput, Seq2SeqListInput

if _TEXT_AVAILABLE:
    from datasets import Dataset
else:
    Dataset = object

# Skip doctests if requirements aren't available
if not _TEXT_TESTING:
    __doctest_skip__ = ["TranslationData", "TranslationData.*"]


class TranslationData(DataModule):
    """The ``TranslationData`` class is a :class:`~flash.core.data.data_module.DataModule` with a set of
    classmethods for loading data for text translation."""

    input_transform_cls = InputTransform

    @classmethod
    def from_csv(
        cls,
        input_field: str,
        target_field: Optional[str] = None,
        train_file: Optional[PATH_TYPE] = None,
        val_file: Optional[PATH_TYPE] = None,
        test_file: Optional[PATH_TYPE] = None,
        predict_file: Optional[PATH_TYPE] = None,
        input_cls: Type[Input] = Seq2SeqCSVInput,
        transform: INPUT_TRANSFORM_TYPE = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "TranslationData":
        """Load the :class:`~flash.text.seq2seq.translation.data.TranslationData` from CSV files containing input
        text snippets and their corresponding target text snippets.

        Input text snippets will be extracted from the ``input_field`` column in the CSV files.
        Target text snippets will be extracted from the ``target_field`` column in the CSV files.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            input_field: The field (column name) in the CSV files containing the input text snippets.
            target_field: The field (column name) in the CSV files containing the target text snippets.
            train_file: The CSV file to use when training.
            val_file: The CSV file to use when validating.
            test_file: The CSV file to use when testing.
            predict_file: The CSV file to use when predicting.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.text.seq2seq.translation.data.TranslationData`.

        Examples
        ________

        The files can be in Comma Separated Values (CSV) format with either a ``.csv`` or ``.txt`` extension.

        .. testsetup::

            >>> import os
            >>> from pandas import DataFrame
            >>> DataFrame.from_dict({
            ...     "pig latin": ["ayay entencesay inyay igpay atinlay", "ellohay orldway"],
            ...     "english": ["a sentence in pig latin", "hello world"],
            ... }).to_csv("train_data.csv", index=False)
            >>> DataFrame.from_dict({
            ...     "pig latin": ["ayay entencesay orfay edictionpray"],
            ... }).to_csv("predict_data.csv", index=False)

        The file ``train_data.csv`` contains the following:

        .. code-block::

            pig latin,english
            ayay entencesay inyay igpay atinlay,a sentence in pig latin
            ellohay orldway,hello world

        The file ``predict_data.csv`` contains the following:

        .. code-block::

            pig latin
            ayay entencesay orfay edictionpray

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.text import TranslationTask, TranslationData
            >>> datamodule = TranslationData.from_csv(
            ...     "pig latin",
            ...     "english",
            ...     train_file="train_data.csv",
            ...     predict_file="predict_data.csv",
            ...     batch_size=2,
            ... )
            >>> model = TranslationTask()
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
            ...     "pig latin": ["ayay entencesay inyay igpay atinlay", "ellohay orldway"],
            ...     "english": ["a sentence in pig latin", "hello world"],
            ... }).to_csv("train_data.tsv", sep="\\t", index=False)
            >>> DataFrame.from_dict({
            ...     "pig latin": ["ayay entencesay orfay edictionpray"],
            ... }).to_csv("predict_data.tsv", sep="\\t", index=False)

        The file ``train_data.tsv`` contains the following:

        .. code-block::

            pig latin                               english
            ayay entencesay inyay igpay atinlay     a sentence in pig latin
            ellohay orldway                         hello world

        The file ``predict_data.tsv`` contains the following:

        .. code-block::

            pig latin
            ayay entencesay orfay edictionpray

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.text import TranslationTask, TranslationData
            >>> datamodule = TranslationData.from_csv(
            ...     "pig latin",
            ...     "english",
            ...     train_file="train_data.tsv",
            ...     predict_file="predict_data.tsv",
            ...     batch_size=2,
            ... )
            >>> model = TranslationTask()
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
            input_key=input_field,
            target_key=target_field,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_file, **ds_kw),
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
        target_field: Optional[str] = None,
        train_file: Optional[PATH_TYPE] = None,
        val_file: Optional[PATH_TYPE] = None,
        test_file: Optional[PATH_TYPE] = None,
        predict_file: Optional[PATH_TYPE] = None,
        input_cls: Type[Input] = Seq2SeqJSONInput,
        transform: INPUT_TRANSFORM_TYPE = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        field: Optional[str] = None,
        **data_module_kwargs: Any,
    ) -> "TranslationData":
        """Load the :class:`~flash.text.seq2seq.translation.data.TranslationData` from JSON files containing input
        text snippets and their corresponding target text snippets.

        Input text snippets will be extracted from the ``input_field`` column in the JSON files.
        Target text snippets will be extracted from the ``target_field`` column in the JSON files.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            input_field: The field (column name) in the JSON objects containing the input text snippets.
            target_field: The field (column name) in the JSON objects containing the target text snippets.
            train_file: The JSON file to use when training.
            val_file: The JSON file to use when validating.
            test_file: The JSON file to use when testing.
            predict_file: The JSON file to use when predicting.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            field: The field that holds the data in the JSON file.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.text.seq2seq.translation.data.TranslationData`.

        Examples
        ________

        .. testsetup::

            >>> import os
            >>> from pandas import DataFrame
            >>> DataFrame.from_dict({
            ...     "pig latin": ["ayay entencesay inyay igpay atinlay", "ellohay orldway"],
            ...     "english": ["a sentence in pig latin", "hello world"],
            ... }).to_json("train_data.json", orient="records", lines=True)
            >>> DataFrame.from_dict({
            ...     "pig latin": ["ayay entencesay orfay edictionpray"],
            ... }).to_json("predict_data.json", orient="records", lines=True)

        The file ``train_data.json`` contains the following:

        .. code-block::

            {"pig latin":"ayay entencesay inyay igpay atinlay","english":"a sentence in pig latin"}
            {"pig latin":"ellohay orldway","english":"hello world"}

        The file ``predict_data.json`` contains the following:

        .. code-block::

            {"pig latin":"ayay entencesay orfay edictionpray"}

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.text import TranslationTask, TranslationData
            >>> datamodule = TranslationData.from_json(
            ...     "pig latin",
            ...     "english",
            ...     train_file="train_data.json",
            ...     predict_file="predict_data.json",
            ...     batch_size=2,
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Downloading...
            >>> model = TranslationTask()
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
            input_key=input_field,
            target_key=target_field,
            field=field,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_file, **ds_kw),
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
        target_field: Optional[str] = None,
        train_hf_dataset: Optional[Dataset] = None,
        val_hf_dataset: Optional[Dataset] = None,
        test_hf_dataset: Optional[Dataset] = None,
        predict_hf_dataset: Optional[Dataset] = None,
        input_cls: Type[Input] = Seq2SeqInputBase,
        transform: INPUT_TRANSFORM_TYPE = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "TranslationData":
        """Load the :class:`~flash.text.seq2seq.translation.data.TranslationData` from Hugging Face ``Dataset``
        objects containing input text snippets and their corresponding target text snippets.

        Input text snippets will be extracted from the ``input_field`` column in the ``Dataset`` objects.
        Target text snippets will be extracted from the ``target_field`` column in the ``Dataset`` objects.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            input_field: The field (column name) in the ``Dataset`` objects containing the input text snippets.
            target_field: The field (column name) in the ``Dataset`` objects containing the target text snippets.
            train_hf_dataset: The ``Dataset`` to use when training.
            val_hf_dataset: The ``Dataset`` to use when validating.
            test_hf_dataset: The ``Dataset`` to use when testing.
            predict_hf_dataset: The ``Dataset`` to use when predicting.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.text.seq2seq.translation.data.TranslationData`.

        Examples
        ________

        .. doctest::

            >>> from datasets import Dataset
            >>> from flash import Trainer
            >>> from flash.text import TranslationTask, TranslationData
            >>> train_data = Dataset.from_dict(
            ...     {
            ...         "pig latin": ["ayay entencesay inyay igpay atinlay", "ellohay orldway"],
            ...         "english": ["a sentence in pig latin", "hello world"],
            ...     }
            ... )
            >>> predict_data = Dataset.from_dict(
            ...     {
            ...         "pig latin": ["ayay entencesay orfay edictionpray"],
            ...     }
            ... )
            >>> datamodule = TranslationData.from_hf_datasets(
            ...     "pig latin",
            ...     "english",
            ...     train_hf_dataset=train_data,
            ...     predict_hf_dataset=predict_data,
            ...     batch_size=2,
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            >>> model = TranslationTask()
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
            input_key=input_field,
            target_key=target_field,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_hf_dataset, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_hf_dataset, **ds_kw),
            input_cls(RunningStage.TESTING, test_hf_dataset, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_hf_dataset, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_lists(
        cls,
        train_data: Optional[List[str]] = None,
        train_targets: Optional[List[str]] = None,
        val_data: Optional[List[str]] = None,
        val_targets: Optional[List[str]] = None,
        test_data: Optional[List[str]] = None,
        test_targets: Optional[List[str]] = None,
        predict_data: Optional[List[str]] = None,
        input_cls: Type[Input] = Seq2SeqListInput,
        transform: INPUT_TRANSFORM_TYPE = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "TranslationData":
        """Load the :class:`~flash.text.seq2seq.translation.data.TranslationData` from lists of input text snippets
        and corresponding lists of target text snippets.

        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_data: The list of input text snippets to use when training.
            train_targets: The list of target text snippets to use when training.
            val_data: The list of input text snippets to use when validating.
            val_targets: The list of target text snippets to use when validating.
            test_data: The list of input text snippets to use when testing.
            test_targets: The list of target text snippets to use when testing.
            predict_data: The list of input text snippets to use when predicting.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.text.seq2seq.translation.data.TranslationData`.

        Examples
        ________

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.text import TranslationTask, TranslationData
            >>> datamodule = TranslationData.from_lists(
            ...     train_data=["ayay entencesay inyay igpay atinlay", "ellohay orldway"],
            ...     train_targets=["a sentence in pig latin", "hello world"],
            ...     predict_data=["ayay entencesay orfay edictionpray"],
            ...     batch_size=2,
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            >>> model = TranslationTask()
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...
        """

        ds_kw = dict()

        return cls(
            input_cls(RunningStage.TRAINING, train_data, train_targets, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_data, val_targets, **ds_kw),
            input_cls(RunningStage.TESTING, test_data, test_targets, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_data, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )
