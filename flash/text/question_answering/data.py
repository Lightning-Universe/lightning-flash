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

# Adapted from:
# https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py
# https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/utils_qa.py

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from pytorch_lightning.trainer.states import RunningStage
from torch import Tensor
from torch.utils.data.sampler import Sampler

import flash
from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_module import DataModule
from flash.core.data.data_source import DataSource, DefaultDataSources
from flash.core.data.process import Postprocess, Preprocess
from flash.core.data.properties import ProcessState
from flash.core.utilities.imports import _TEXT_AVAILABLE, requires_extras
from flash.text.classification.data import TextDeserializer

if _TEXT_AVAILABLE:
    import datasets
    from datasets import Dataset, DatasetDict, load_dataset
    from transformers import AutoTokenizer, default_data_collator

# TODO:
#   1) Updates needed
#       i) Update the `QuestionAnsweringPostprocess` class according to the example
#       ii) Actual implementation of all methods and classes.
#       iii) `SQuADDataSource` might also need an update.


class QuestionAnsweringDataSource(DataSource):

    @requires_extras("text")
    def __init__(
        self,
        backbone: str,
        max_source_length: int = 128,
        max_target_length: int = 128,
        padding: Union[str, bool] = 'max_length',
    ):
        super().__init__()

        self.backbone = backbone
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.padding = padding

        # Setup global pre-processing requirements
        #   pad_on_right, doc_stride, etc. are remaining
        self._question_column_name = "question"
        self._context_column_name = "context"
        self._answer_column_name = "answer"
        self._doc_stride = 128

    def _tokenize_fn(self, samples: Any) -> Callable:

        # Setup the tokenizing call here
        #   tokenized_samples = self.tokenizer(...params)
        self.pad_on_right = self.tokenizer.padding_side == "right"

        tokenized_samples = self.tokenizer(
            samples[self._question_column_name if self.pad_on_right else self._context_column_name],
            samples[self._context_column_name if self.pad_on_right else self._question_column_name],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_source_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=self.padding
        )

        stage = self._running_stage.value

        if stage == RunningStage.TRAINING:
            # Preprocess function for training
            tokenized_samples = self._prepare_train_features(samples, tokenized_samples)

        if self._running_stage.evaluating() or stage == RunningStage.PREDICTING:
            # Preprocess function for eval or predict
            tokenized_samples = self._prepare_val_features(samples, tokenized_samples)

        return tokenized_samples

    def _prepare_train_features(self, samples: Any, tokenized_samples: Any):
        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_samples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_samples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_samples["start_positions"] = []
        tokenized_samples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_samples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_samples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = samples[self._answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_samples["start_positions"].append(cls_index)
                tokenized_samples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if self.pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if self.pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_samples["start_positions"].append(cls_index)
                    tokenized_samples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_samples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_samples["end_positions"].append(token_end_index + 1)

        return tokenized_samples

    def _prepare_val_features(self, samples: Any, tokenized_samples: Any):
        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_samples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_samples["example_id"] = []

        for i in range(len(tokenized_samples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_samples.sequence_ids(i)
            context_index = 1 if self.pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_samples["example_id"].append(samples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_samples["offset_mapping"][i] = [(o if sequence_ids[k] == context_index else None)
                                                      for k, o in enumerate(tokenized_samples["offset_mapping"][i])]

        return tokenized_samples

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("tokenizer")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True)

    @property
    def question_column_name(self) -> str:
        return self._question_column_name

    @property
    def context_column_name(self) -> str:
        return self._context_column_name

    @property
    def answer_column_name(self) -> str:
        return self._answer_column_name

    @property
    def doc_stride(self) -> str:
        return self._doc_stride

    @question_column_name.setter
    def question_column_name(self, value: str) -> None:
        self._question_column_name = value

    @context_column_name.setter
    def context_column_name(self, value: str) -> None:
        self._context_column_name = value

    @answer_column_name.setter
    def answer_column_name(self, value: str) -> None:
        self._answer_column_name = value

    @doc_stride.setter
    def doc_stride(self, value: int) -> None:
        self._doc_stride = value


class QuestionAnsweringFileDataSource(QuestionAnsweringDataSource):

    def __init__(
        self,
        filetype: str,
        backbone: str,
        max_source_length: int = 128,
        max_target_length: int = 128,
        padding: Union[str, bool] = 'max_length',
    ):
        super().__init__(backbone, max_source_length, max_target_length, padding)

        self.filetype = filetype

    def load_data(self, data: Any, columns: List[str] = None) -> 'datasets.Dataset':
        if columns is None:
            columns = ["input_ids", "attention_mask", "labels"]
        if self.filetype == 'json':
            file, question_column_name, context_column_name, answer_column_name, doc_stride, field = data
        else:
            file, question_column_name, context_column_name, answer_column_name, doc_stride = data

        self.question_column_name(question_column_name)
        self.context_column_name(context_column_name)
        self.answer_column_name(answer_column_name)
        self.doc_stride(doc_stride)

        data_files = {}
        stage = self._running_stage.value
        data_files[stage] = str(file)

        # FLASH_TESTING is set in the CI to run faster.
        if flash._IS_TESTING:
            try:
                if self.filetype == 'json' and field is not None:
                    dataset_dict = DatasetDict({
                        stage: load_dataset(self.filetype, data_files=data_files, split=[f'{stage}[:20]'],
                                            field=field)[0]
                    })
                else:
                    dataset_dict = DatasetDict({
                        stage: load_dataset(self.filetype, data_files=data_files, split=[f'{stage}[:20]'])[0]
                    })
            except Exception:
                if self.filetype == 'json' and field is not None:
                    dataset_dict = load_dataset(self.filetype, data_files=data_files, field=field)
                else:
                    dataset_dict = load_dataset(self.filetype, data_files=data_files)
        else:
            if self.filetype == 'json' and field is not None:
                dataset_dict = load_dataset(self.filetype, data_files=data_files, field=field)
            else:
                dataset_dict = load_dataset(self.filetype, data_files=data_files)

        dataset_dict = dataset_dict.map(self._tokenize_fn, batched=True)
        dataset_dict.set_format(columns=columns)
        return dataset_dict[stage]

    def predict_load_data(self, data: Any) -> Union['datasets.Dataset', List[Dict[str, torch.Tensor]]]:
        return self.load_data(data, columns=["input_ids", "attention_mask"])

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("tokenizer")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True)


class QuestionAnsweringCSVDataSource(QuestionAnsweringFileDataSource):

    def __init__(
        self,
        backbone: str,
        max_source_length: int = 128,
        max_target_length: int = 128,
        padding: Union[str, bool] = 'max_length',
    ):
        super().__init__(
            "csv",
            backbone,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
        )

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("tokenizer")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True)


class QuestionAnsweringJSONDataSource(QuestionAnsweringFileDataSource):

    def __init__(
        self,
        backbone: str,
        max_source_length: int = 128,
        max_target_length: int = 128,
        padding: Union[str, bool] = 'max_length',
    ):
        super().__init__(
            "json",
            backbone,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
        )

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("tokenizer")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True)


class QuestionAnsweringDictionaryDataSource(QuestionAnsweringDataSource):

    def load_data(self, data: Any, columns: List[str] = None) -> 'datasets.Dataset':
        if columns is None:
            columns = ["input_ids", "attention_mask", "labels"]
            if self._running_stage.value == RunningStage.PREDICTING:
                columns.remove("labels")

        stage = self._running_stage.value

        dataset_dict = DatasetDict({stage: Dataset.from_dict(data)})
        dataset_dict = dataset_dict.map(self._tokenize_fn, batched=True)

        dataset_dict.set_format(columns=columns)
        return dataset_dict[stage]

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("tokenizer")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True)


class SQuADDataSource(QuestionAnsweringDataSource):

    def load_data(self, data: str, dataset: Optional[Any] = None) -> 'datasets.Dataset':
        stage = self._running_stage.value

        file_path, doc_stride = data

        self.question_column_name("question")
        self.context_column_name("context")
        self.answer_column_name("answer")
        self.doc_stride(doc_stride)

        path = Path(file_path)
        with open(path, 'rb') as f:
            squad_v_2_dict = json.load(f)

        titles = []
        contexts = []
        questions = []
        answers = []
        for topic in squad_v_2_dict['data']:
            title = topic["title"]
            for comprehension in topic['paragraphs']:
                context = comprehension['context']
                for q_a_pair in comprehension['qas']:
                    question = q_a_pair['question']
                    for answer in q_a_pair['answers']:
                        answer_text = answer['text']
                        answer_start = answer['answer_start']

                        titles.append(title)
                        contexts.append(context)
                        questions.append(question)
                        answers.append(dict(text=answer_text, answer_start=answer_start))

        dataset_dict = DatasetDict({
            stage: Dataset.from_dict({
                "title": titles,
                "context": contexts,
                "question": questions,
                "answer": answers
            })
        })

        dataset_dict = dataset_dict.map(self._tokenize_fn, batched=True)

        return dataset_dict[stage]


@dataclass(unsafe_hash=True, frozen=True)
class QuestionAnsweringBackboneState(ProcessState):
    """The ``QuestionAnsweringBackboneState`` stores the backbone in use by the
    :class:`~flash.text.question_answering.data.QuestionAnsweringPreprocess`
    """

    backbone: str


class QuestionAnsweringPreprocess(Preprocess):

    @requires_extras("text")
    def __init__(
        self,
        question_column_name: Optional[str] = None,
        context_column_name: Optional[str] = None,
        answer_column_name: Optional[str] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        backbone: str = "t5-small",
        max_source_length: int = 128,
        max_target_length: int = 128,
        padding: Union[str, bool] = 'max_length'
    ):
        self.backbone = backbone
        self.max_target_length = max_target_length
        self.max_source_length = max_source_length
        self.padding = padding

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources={
                DefaultDataSources.CSV: QuestionAnsweringCSVDataSource(
                    self.backbone,
                    max_source_length=max_source_length,
                    max_target_length=max_target_length,
                    padding=padding,
                    question_column_name=question_column_name,
                    context_column_name=context_column_name,
                    answer_column_name=answer_column_name,
                ),
                DefaultDataSources.JSON: QuestionAnsweringJSONDataSource(
                    self.backbone,
                    max_source_length=max_source_length,
                    max_target_length=max_target_length,
                    padding=padding,
                    question_column_name=question_column_name,
                    context_column_name=context_column_name,
                    answer_column_name=answer_column_name,
                ),
                "dict": QuestionAnsweringDictionaryDataSource(
                    self.backbone,
                    max_source_length=max_source_length,
                    max_target_length=max_target_length,
                    padding=padding,
                    question_column_name=question_column_name,
                    context_column_name=context_column_name,
                    answer_column_name=answer_column_name,
                ),
                "squad_v2": SQuADDataSource(
                    self.backbone,
                    max_source_length=max_source_length,
                    max_target_length=max_target_length,
                    padding=padding,
                )
            },
            default_data_source="dict",
            deserializer=TextDeserializer(backbone, max_source_length)
        )

        self.set_state(QuestionAnsweringBackboneState(self.backbone))

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            **self.transforms,
            "backbone": self.backbone,
            "max_source_length": self.max_source_length,
            "max_target_length": self.max_target_length,
            "padding": self.padding,
        }

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool):
        return cls(**state_dict)

    def collate(self, samples: Any) -> Tensor:
        """Override to convert a set of samples to a batch"""
        return default_data_collator(samples)


class QuestionAnsweringPostprocess(Postprocess):

    @requires_extras("text")
    def __init__(self):
        super().__init__()

        self._backbone = None
        self._tokenizer = None

    @property
    def backbone(self):
        backbone_state = self.get_state(QuestionAnsweringBackboneState)
        if backbone_state is not None:
            return backbone_state.backbone

    @property
    def tokenizer(self):
        if self.backbone is not None and self.backbone != self._backbone:
            self._tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True)
            self._backbone = self.backbone
        return self._tokenizer

    def uncollate(self, generated_tokens: Any) -> Any:
        pred_str = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        pred_str = [str.strip(s) for s in pred_str]
        return pred_str

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("_tokenizer")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True)


class QuestionAnsweringData(DataModule):
    """Data module for QuestionAnswering task."""

    preprocess_cls = QuestionAnsweringPreprocess
    postprocess_cls = QuestionAnsweringPostprocess

    @classmethod
    def from_json(
        cls,
        question_column_name: Optional[str] = "question",
        context_column_name: Optional[str] = "context",
        answer_column_name: Optional[str] = "answer",
        doc_stride: Optional[int] = 128,
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
        sampler: Optional[Sampler] = None,
        field: Optional[str] = None,
        **preprocess_kwargs: Any,
    ) -> 'DataModule':
        """Creates a :class:`~flash.text.question_answering.data.QuestionAnsweringData` object from the given
        JSON files using the :class:`~flash.text.question_answering.data.QuestionAnsweringDataSource`
        of name :attr:`~flash.core.data.data_source.DefaultDataSources.JSON` from the passed or
        constructed :class:`~flash.text.question_answering.data.QuestionAnsweringPreprocess`.

        Args:
            question_column_name: The field in the JSON objects to use for the question.
            context_column_name: The field in the JSON objects to use for the context.
            answer_column_name: The field in the JSON objects to use for the answer.
            doc_stride: The amount of stride to be taken between chunks when splitting up a long document into chunks.
            train_file: The JSON file containing the training data.
            val_file: The JSON file containing the validation data.
            test_file: The JSON file containing the testing data.
            predict_file: The JSON file containing the data to use when predicting.
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
            sampler: The ``sampler`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            field: To specify the field that holds the data in the JSON file.
            preprocess_kwargs: Additional keyword arguments to use when constructing the preprocess. Will only be used
                if ``preprocess = None``.

        Returns:
            The constructed data module.

        Examples::

            data_module = DataModule.from_json(
                question_column_name="question",
                context_column_name="context",
                answer_column_name="answer",
                doc_stride=128,
                train_file="train_data.json",
                train_transform={
                    "to_tensor_transform": torch.as_tensor,
                },
            )
        """
        return cls.from_data_source(
            DefaultDataSources.JSON,
            (train_file, question_column_name, context_column_name, answer_column_name, doc_stride, field),
            (val_file, question_column_name, context_column_name, answer_column_name, doc_stride, field),
            (test_file, question_column_name, context_column_name, answer_column_name, doc_stride, field),
            (predict_file, question_column_name, context_column_name, answer_column_name, doc_stride, field),
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            **preprocess_kwargs,
        )

    @classmethod
    def from_csv(
        cls,
        question_column_name: Optional[str] = "question",
        context_column_name: Optional[str] = "context",
        answer_column_name: Optional[str] = "answer",
        doc_stride: Optional[int] = 128,
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
        sampler: Optional[Sampler] = None,
        **preprocess_kwargs: Any,
    ) -> 'DataModule':
        """Creates a :class:`~flash.text.question_answering.data.QuestionAnsweringData` object from the given
        JSON files using the :class:`~flash.text.question_answering.data.QuestionAnsweringDataSource`
        of name :attr:`~flash.core.data.data_source.DefaultDataSources.CSV` from the passed or
        constructed :class:`~flash.text.question_answering.data.QuestionAnsweringPreprocess`.

        Args:
            question_column_name: The field in the JSON objects to use for the question.
            context_column_name: The field in the JSON objects to use for the context.
            answer_column_name: The field in the JSON objects to use for the answer.
            doc_stride: The amount of stride to be taken between chunks when splitting up a long document into chunks.
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
            sampler: The ``sampler`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            preprocess_kwargs: Additional keyword arguments to use when constructing the preprocess. Will only be used
                if ``preprocess = None``.

        Returns:
            The constructed data module.

        Examples::

            data_module = DataModule.from_csv(
                question_column_name="question",
                context_column_name="context",
                answer_column_name="answer",
                doc_stride=128,
                train_file="train_data.csv",
                train_transform={
                    "to_tensor_transform": torch.as_tensor,
                },
            )
        """
        return cls.from_data_source(
            DefaultDataSources.CSV,
            (train_file, question_column_name, context_column_name, answer_column_name, doc_stride),
            (val_file, question_column_name, context_column_name, answer_column_name, doc_stride),
            (test_file, question_column_name, context_column_name, answer_column_name, doc_stride),
            (predict_file, question_column_name, context_column_name, answer_column_name, doc_stride),
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            **preprocess_kwargs,
        )

    @classmethod
    def from_squad_v2(
        cls,
        doc_stride: Optional[int] = 128,
        train_file: Optional[str] = None,
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        **preprocess_kwargs: Any,
    ):
        """Creates a :class:`~flash.text.question_answering.data.QuestionAnsweringData` object from the given
        data JSON files in the SQuAD2.0 format.

        Args:
            doc_stride: The amount of stride to be taken between chunks when splitting up a long document into chunks.
            train_file: The JSON file containing the training data.
            val_file: The JSON file containing the validation data.
            test_file: The JSON file containing the testing data.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
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

            data_module = QuestionAnsweringData.from_squad_v2(
                doc_stride=128,
                train_file="train.json",
            )
        """
        return cls.from_data_source(
            "squad_v2",
            (train_file, doc_stride),
            (val_file, doc_stride),
            (test_file, doc_stride),
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            **preprocess_kwargs,
        )
