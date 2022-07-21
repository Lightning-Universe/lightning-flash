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
from typing import Any, Dict, Optional, Type

from flash.core.data.data_module import DataModule
from flash.core.data.io.input import Input
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.utilities.paths import PATH_TYPE
from flash.core.utilities.imports import _TEXT_AVAILABLE
from flash.core.utilities.stages import RunningStage
from flash.core.utilities.types import INPUT_TRANSFORM_TYPE
from flash.text.question_answering.input import (
    QuestionAnsweringCSVInput,
    QuestionAnsweringDictionaryInput,
    QuestionAnsweringJSONInput,
    QuestionAnsweringSQuADInput,
)

# Skip doctests if requirements aren't available
if not _TEXT_AVAILABLE:
    __doctest_skip__ = ["QuestionAnsweringData", "QuestionAnsweringData.*"]


class QuestionAnsweringData(DataModule):
    """The ``QuestionAnsweringData`` class is a :class:`~flash.core.data.data_module.DataModule` with a set of
    classmethods for loading data for extractive question answering."""

    input_transform_cls = InputTransform

    @classmethod
    def from_csv(
        cls,
        train_file: Optional[PATH_TYPE] = None,
        val_file: Optional[PATH_TYPE] = None,
        test_file: Optional[PATH_TYPE] = None,
        predict_file: Optional[PATH_TYPE] = None,
        input_cls: Type[Input] = QuestionAnsweringCSVInput,
        transform: INPUT_TRANSFORM_TYPE = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        question_column_name: str = "question",
        context_column_name: str = "context",
        answer_column_name: str = "answer",
        **data_module_kwargs: Any,
    ) -> "QuestionAnsweringData":
        """Load the :class:`~flash.text.question_answering.data.QuestionAnsweringData` from CSV files containing
        questions, contexts and their corresponding answers.

        Question snippets will be extracted from the ``question_column_name`` column in the CSV files.
        Context snippets will be extracted from the ``context_column_name`` column in the CSV files.
        Answer snippets will be extracted from the ``answer_column_name`` column in the CSV files.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_file: The CSV file containing the training data.
            val_file: The CSV file containing the validation data.
            test_file: The CSV file containing the testing data.
            predict_file: The CSV file containing the data to use when predicting.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            question_column_name: The key in the JSON file to recognize the question field.
            context_column_name: The key in the JSON file to recognize the context field.
            answer_column_name: The key in the JSON file to recognize the answer field.

        Returns:
            The constructed :class:`~flash.text.question_answering.data.QuestionAnsweringData`.

        Examples
        ________

        The files can be in Comma Separated Values (CSV) format with either a ``.csv`` or ``.txt`` extension.

        .. testsetup::

            >>> import os
            >>> from pandas import DataFrame
            >>> DataFrame.from_dict({
            ...     "id": ["1", "2", "3"],
            ...     "context": [
            ...         "I am three years old",
            ...         "I am six feet tall",
            ...         "I am eight years old",
            ...     ],
            ...     "question": ["How old are you?", "How tall are you?", "How old are you?"],
            ...     "answer_text": ["three", "six", "eight"],
            ...     "answer_start": [0, 0, 0],
            ... }).to_csv("train_data.csv", index=False)
            >>> DataFrame.from_dict({
            ...     "id": ["4"],
            ...     "context": [
            ...         "I am five feet tall",
            ...     ],
            ...     "question": [
            ...         "How tall are you?",
            ...     ],
            ... }).to_csv("predict_data.csv", index=False)

        The file ``train_data.csv`` contains the following:

        .. code-block::

            id,context,question,answer_text,answer_start
            1,I am three years old,How old are you?,three,0
            2,I am six feet tall,How tall are you?,six,0
            3,I am eight years old,How old are you?,eight,0

        The file ``predict_data.csv`` contains the following:

        .. code-block::

            id,context,question
            4,I am five feet tall,How tall are you?

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.text import QuestionAnsweringData, QuestionAnsweringTask
            >>> datamodule = QuestionAnsweringData.from_csv(
            ...     train_file="train_data.csv",
            ...     predict_file="predict_data.csv",
            ...     batch_size=2,
            ... )
            >>> model = QuestionAnsweringTask(max_source_length=32, max_target_length=32)
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
            ...     "id": ["1", "2", "3"],
            ...     "context": [
            ...         "I am three years old",
            ...         "I am six feet tall",
            ...         "I am eight years old",
            ...     ],
            ...     "question": ["How old are you?", "How tall are you?", "How old are you?"],
            ...     "answer_text": ["three", "six", "eight"],
            ...     "answer_start": [0, 0, 0],
            ... }).to_csv("train_data.tsv", sep="\\t", index=False)
            >>> DataFrame.from_dict({
            ...     "id": ["4"],
            ...     "context": [
            ...         "I am five feet tall",
            ...     ],
            ...     "question": [
            ...         "How tall are you?",
            ...     ],
            ... }).to_csv("predict_data.tsv", sep="\\t", index=False)

        The file ``train_data.tsv`` contains the following:

        .. code-block::

            id  context                 question            answer_text answer_start
            1   I am three years old    How old are you?    three       0
            2   I am six feet tall	    How tall are you?   six	        0
            3   I am eight years old    How old are you?    eight       0

        The file ``predict_data.tsv`` contains the following:

        .. code-block::

            id  context             question
            4   I am five feet tall How tall are you?

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.text import QuestionAnsweringData, QuestionAnsweringTask
            >>> datamodule = QuestionAnsweringData.from_csv(
            ...     train_file="train_data.tsv",
            ...     predict_file="predict_data.tsv",
            ...     batch_size=2,
            ... )
            >>> model = QuestionAnsweringTask(max_source_length=32, max_target_length=32)
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
            question_column_name=question_column_name,
            context_column_name=context_column_name,
            answer_column_name=answer_column_name,
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
        train_file: Optional[PATH_TYPE] = None,
        val_file: Optional[PATH_TYPE] = None,
        test_file: Optional[PATH_TYPE] = None,
        predict_file: Optional[PATH_TYPE] = None,
        input_cls: Type[Input] = QuestionAnsweringJSONInput,
        transform: INPUT_TRANSFORM_TYPE = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        field: Optional[str] = None,
        question_column_name: str = "question",
        context_column_name: str = "context",
        answer_column_name: str = "answer",
        **data_module_kwargs: Any,
    ) -> "QuestionAnsweringData":
        """Load the :class:`~flash.text.question_answering.data.QuestionAnsweringData` from JSON files containing
        questions, contexts and their corresponding answers.

        Question snippets will be extracted from the ``question_column_name`` column in the JSON files.
        Context snippets will be extracted from the ``context_column_name`` column in the JSON files.
        Answer snippets will be extracted from the ``answer_column_name`` column in the JSON files.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_file: The JSON file containing the training data.
            val_file: The JSON file containing the validation data.
            test_file: The JSON file containing the testing data.
            predict_file: The JSON file containing the data to use when predicting.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            field: The field that holds the data in the JSON file.
            question_column_name: The key in the JSON file to recognize the question field.
            context_column_name: The key in the JSON file to recognize the context field.
            answer_column_name: The key in the JSON file to recognize the answer field.

        Returns:
            The constructed :class:`~flash.text.question_answering.data.QuestionAnsweringData`.

        Examples
        ________

        .. testsetup::

            >>> import os
            >>> from pandas import DataFrame
            >>> DataFrame.from_dict({
            ...     "id": ["12345", "12346", "12347", "12348"],
            ...     "context": [
            ...         "this is an answer one. this is a context one",
            ...         "this is an answer two. this is a context two",
            ...         "this is an answer three. this is a context three",
            ...         "this is an answer four. this is a context four",
            ...     ],
            ...     "question": [
            ...         "this is a question one",
            ...         "this is a question two",
            ...         "this is a question three",
            ...         "this is a question four",
            ...     ],
            ...     "answer_text": [
            ...         "this is an answer one",
            ...         "this is an answer two",
            ...         "this is an answer three",
            ...         "this is an answer four",
            ...     ],
            ...     "answer_start": [0, 0, 0, 0],
            ... }).to_json("train_data.json", orient="records", lines=True)
            >>> DataFrame.from_dict({
            ...     "id": ["12349", "12350"],
            ...     "context": [
            ...         "this is an answer five. this is a context five",
            ...         "this is an answer six. this is a context six",
            ...     ],
            ...     "question": [
            ...         "this is a question five",
            ...         "this is a question six",
            ...     ],
            ... }).to_json("predict_data.json", orient="records", lines=True)

        The file ``train_data.json`` contains the following:

        .. code-block::

            {"id":"12345","context":"this is an answer one. this is a context one","question":"this is a question one",
            "answer_text":"this is an answer one","answer_start":0}
            {"id":"12346","context":"this is an answer two. this is a context two","question":"this is a question two",
            "answer_text":"this is an answer two","answer_start":0}
            {"id":"12347","context":"this is an answer three. this is a context three","question":"this is a question
             three","answer_text":"this is an answer three","answer_start":0}
            {"id":"12348","context":"this is an answer four. this is a context four","question":"this is a question
             four","answer_text":"this is an answer four","answer_start":0}


        The file ``predict_data.json`` contains the following:

        .. code-block::

            {"id":"12349","context":"this is an answer five. this is a context five","question":"this is a question
             five"}
            {"id":"12350","context":"this is an answer six. this is a context six","question":"this is a question
             six"}

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.text import QuestionAnsweringData, QuestionAnsweringTask
            >>> datamodule = QuestionAnsweringData.from_json(
            ...     train_file="train_data.json",
            ...     predict_file="predict_data.json",
            ...     batch_size=2,
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Downloading...
            >>> model = QuestionAnsweringTask(max_source_length=32, max_target_length=32)
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
            field=field,
            question_column_name=question_column_name,
            context_column_name=context_column_name,
            answer_column_name=answer_column_name,
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
    def from_squad_v2(
        cls,
        train_file: Optional[str] = None,
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        predict_file: Optional[str] = None,
        input_cls: Type[Input] = QuestionAnsweringSQuADInput,
        transform: INPUT_TRANSFORM_TYPE = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        question_column_name: str = "question",
        context_column_name: str = "context",
        answer_column_name: str = "answer",
        **data_module_kwargs: Any,
    ) -> "QuestionAnsweringData":
        """Load the :class:`~flash.text.question_answering.data.QuestionAnsweringData` from JSON files containing
        questions, contexts and their corresponding answers in the SQuAD2.0 format.

        Question snippets will be extracted from the ``question_column_name`` column in the JSON files.
        Context snippets will be extracted from the ``context_column_name`` column in the JSON files.
        Answer snippets will be extracted from the ``answer_column_name`` column in the JSON files.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_file: The JSON file containing the training data.
            val_file: The JSON file containing the validation data.
            test_file: The JSON file containing the testing data.
            predict_file: The JSON file containing the predict data.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            question_column_name: The key in the JSON file to recognize the question field.
            context_column_name: The key in the JSON file to recognize the context field.
            answer_column_name: The key in the JSON file to recognize the answer field.

        Returns:
            The constructed data module.

        Examples
        ________

        .. testsetup::

            >>> import json
            >>> from pathlib import Path
            >>> train_data = Path("train_data.json")
            >>> predict_data = Path("predict_data.json")
            >>> _ = train_data.write_text(
            ...     json.dumps(
            ...         {
            ...             "version": "v2.0",
            ...             "data": [
            ...                 {
            ...                     "title": "ExampleSet1",
            ...                     "paragraphs": [
            ...                         {
            ...                             "qas": [
            ...                                 {
            ...                                     "question": "this is a question one",
            ...                                     "id": "12345",
            ...                                     "answers": [{"text": "this is an answer one", "answer_start": 0}],
            ...                                     "is_impossible": False,
            ...                                 }
            ...                             ],
            ...                             "context": "this is an answer one. this is a context one",
            ...                         },
            ...                         {
            ...                             "qas": [
            ...                                 {
            ...                                     "question": "this is a question two",
            ...                                     "id": "12346",
            ...                                     "answers": [{"text": "this is an answer two", "answer_start": 0}],
            ...                                     "is_impossible": False,
            ...                                 }
            ...                             ],
            ...                             "context": "this is an answer two. this is a context two",
            ...                         },
            ...                     ],
            ...                 },
            ...                 {
            ...                     "title": "ExampleSet2",
            ...                     "paragraphs": [
            ...                         {
            ...                             "qas": [
            ...                                 {
            ...                                     "question": "this is a question three",
            ...                                     "id": "12347",
            ...                                     "answers": [{"text": "this is an answer three", "answer_start": 0}],
            ...                                     "is_impossible": False,
            ...                                 }
            ...                             ],
            ...                             "context": "this is an answer three. this is a context three",
            ...                         },
            ...                         {
            ...                             "qas": [
            ...                                 {
            ...                                     "question": "this is a question four",
            ...                                     "id": "12348",
            ...                                     "answers": [{"text": "this is an answer four", "answer_start": 0}],
            ...                                     "is_impossible": False,
            ...                                 }
            ...                             ],
            ...                             "context": "this is an answer four. this is a context four",
            ...                         },
            ...                     ],
            ...                 },
            ...             ],
            ...         }
            ...     )
            ... )
            >>> _ = predict_data.write_text(
            ...     json.dumps(
            ...         {
            ...             "version": "v2.0",
            ...             "data": [
            ...                 {
            ...                     "title": "ExampleSet3",
            ...                     "paragraphs": [
            ...                         {
            ...                             "qas": [
            ...                                 {
            ...                                     "question": "this is a question five",
            ...                                     "id": "12349",
            ...                                     "is_impossible": False,
            ...                                 }
            ...                             ],
            ...                             "context": "this is an answer five. this is a context five",
            ...                         },
            ...                         {
            ...                             "qas": [
            ...                                 {
            ...                                     "question": "this is a question six",
            ...                                     "id": "12350",
            ...                                     "is_impossible": False,
            ...                                 }
            ...                             ],
            ...                             "context": "this is an answer six. this is a context six",
            ...                         },
            ...                     ],
            ...                 }
            ...             ],
            ...         }
            ...     )
            ... )

        The file ``train_data.json`` contains the following:

        .. code-block::

            {
                "version": "v2.0",
                "data": [
                    {
                        "title": "ExampleSet1",
                        "paragraphs": [
                            {
                                "qas": [
                                    {
                                        "question": "this is a question one",
                                        "id": "12345",
                                        "answers": [{"text": "this is an answer one", "answer_start": 0}],
                                        "is_impossible": false
                                    }
                                ],
                                "context": "this is an answer one. this is a context one"
                            }, {
                                "qas": [
                                    {
                                        "question": "this is a question two",
                                        "id": "12346",
                                        "answers": [{"text": "this is an answer two", "answer_start": 0}],
                                        "is_impossible": false
                                    }
                                ],
                                "context": "this is an answer two. this is a context two"
                            }
                        ]
                    }, {
                        "title": "ExampleSet2",
                        "paragraphs": [
                            {
                                "qas": [
                                    {
                                        "question": "this is a question three",
                                        "id": "12347",
                                        "answers": [{"text": "this is an answer three", "answer_start": 0}],
                                        "is_impossible": false
                                    }
                                ],
                                "context": "this is an answer three. this is a context three"
                            }, {
                                "qas": [
                                    {
                                        "question": "this is a question four",
                                        "id": "12348",
                                        "answers": [{"text": "this is an answer four", "answer_start": 0}],
                                        "is_impossible": false
                                    }
                                ],
                                "context": "this is an answer four. this is a context four"
                            }
                        ]
                    }
                ]
            }


        The file ``predict_data.json`` contains the following:

        .. code-block::

            {
                "version": "v2.0",
                "data": [
                    {
                        "title": "ExampleSet3",
                        "paragraphs": [
                            {
                                "qas": [
                                    {
                                        "question": "this is a question five",
                                        "id": "12349",
                                        "is_impossible": false
                                    }
                                ],
                                "context": "this is an answer five. this is a context five"
                            }, {
                                "qas": [
                                    {
                                        "question": "this is a question six",
                                        "id": "12350",
                                        "is_impossible": false
                                    }
                                ],
                                "context": "this is an answer six. this is a context six"
                            }
                        ]
                    }
                ]
            }

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.text import QuestionAnsweringData, QuestionAnsweringTask
            >>> datamodule = QuestionAnsweringData.from_squad_v2(
            ...     train_file="train_data.json",
            ...     predict_file="predict_data.json",
            ...     batch_size=2,
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            >>> model = QuestionAnsweringTask(max_source_length=32, max_target_length=32)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> import os
            >>> os.remove("train_data.json")
            >>> os.remove("predict_data.json")
        """

        ds_kw = dict(
            question_column_name=question_column_name,
            context_column_name=context_column_name,
            answer_column_name=answer_column_name,
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
    def from_dicts(
        cls,
        train_data: Optional[Dict[str, Any]] = None,
        val_data: Optional[Dict[str, Any]] = None,
        test_data: Optional[Dict[str, Any]] = None,
        predict_data: Optional[Dict[str, Any]] = None,
        input_cls: Type[Input] = QuestionAnsweringDictionaryInput,
        transform: INPUT_TRANSFORM_TYPE = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        question_column_name: str = "question",
        context_column_name: str = "context",
        answer_column_name: str = "answer",
        **data_module_kwargs: Any,
    ) -> "QuestionAnsweringData":
        """Load the :class:`~flash.text.question_answering.data.QuestionAnsweringData` from Python dictionary
        objects containing questions, contexts and their corresponding answers.

        Question snippets will be extracted from the ``question_column_name`` field in the dictionaries.
        Context snippets will be extracted from the ``context_column_name`` field in the dictionaries.
        Answer snippets will be extracted from the ``answer_column_name`` field in the dictionaries.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_data: The dictionary containing the training data.
            val_data: The dictionary containing the validation data.
            test_data: The dictionary containing the testing data.
            predict_data: The dictionary containing the data to use when predicting.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            question_column_name: The key in the JSON file to recognize the question field.
            context_column_name: The key in the JSON file to recognize the context field.
            answer_column_name: The key in the JSON file to recognize the answer field.

        Returns:
            The constructed :class:`~flash.text.question_answering.data.QuestionAnsweringData`.

        Examples
        ________

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.text import QuestionAnsweringData, QuestionAnsweringTask
            >>> train_data = {
            ...     "id": ["12345", "12346", "12347", "12348"],
            ...     "context": [
            ...         "this is an answer one. this is a context one",
            ...         "this is an answer two. this is a context two",
            ...         "this is an answer three. this is a context three",
            ...         "this is an answer four. this is a context four",
            ...     ],
            ...     "question": [
            ...         "this is a question one",
            ...         "this is a question two",
            ...         "this is a question three",
            ...         "this is a question four",
            ...     ],
            ...     "answer_text": [
            ...         "this is an answer one",
            ...         "this is an answer two",
            ...         "this is an answer three",
            ...         "this is an answer four",
            ...     ],
            ...     "answer_start": [0, 0, 0, 0],
            ... }
            >>> predict_data = {
            ...     "id": ["12349", "12350"],
            ...     "context": [
            ...         "this is an answer five. this is a context five",
            ...         "this is an answer six. this is a context six",
            ...     ],
            ...     "question": [
            ...         "this is a question five",
            ...         "this is a question six",
            ...     ],
            ... }
            >>> datamodule = QuestionAnsweringData.from_dicts(
            ...     train_data=train_data,
            ...     predict_data=predict_data,
            ...     batch_size=2,
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            >>> model = QuestionAnsweringTask(max_source_length=32, max_target_length=32)
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
            question_column_name=question_column_name,
            context_column_name=context_column_name,
            answer_column_name=answer_column_name,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_data, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_data, **ds_kw),
            input_cls(RunningStage.TESTING, test_data, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_data, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )
