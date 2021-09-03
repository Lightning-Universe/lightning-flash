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
# https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa_no_trainer.py
# https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/utils_qa.py

import collections
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
from flash.core.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources
from flash.core.data.process import Postprocess, Preprocess
from flash.core.data.properties import ProcessState
from flash.core.utilities.imports import _TEXT_AVAILABLE, requires_extras

if _TEXT_AVAILABLE:
    import datasets
    from datasets import Dataset, DatasetDict, load_dataset
    from transformers import AutoTokenizer, DataCollatorWithPadding, default_data_collator


class QuestionAnsweringDataSource(DataSource):
    @requires_extras("text")
    def __init__(
        self,
        backbone: str,
        max_source_length: int = 384,
        max_target_length: int = 30,
        padding: Union[str, bool] = "max_length",
        question_column_name: str = "question",
        context_column_name: str = "context",
        answer_column_name: str = "answer",
        doc_stride: int = 128,
    ):
        super().__init__()

        self.backbone = backbone
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.padding = padding

        # Setup global pre-processing requirements
        self.pad_on_right = self.tokenizer.padding_side == "right"
        self._question_column_name = question_column_name
        self._context_column_name = context_column_name
        self._answer_column_name = answer_column_name
        self._doc_stride = doc_stride

    def _tokenize_fn(self, samples: Any) -> Callable:
        stage = self._running_stage.value

        samples[self.question_column_name] = [q.lstrip() for q in samples[self.question_column_name]]

        tokenized_samples = self.tokenizer(
            samples[self._question_column_name if self.pad_on_right else self._context_column_name],
            samples[self._context_column_name if self.pad_on_right else self._question_column_name],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_source_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=self.padding,
        )

        if stage == RunningStage.TRAINING:
            # Preprocess function for training
            tokenized_samples = self._prepare_train_features(samples, tokenized_samples)
        elif self._running_stage.evaluating or stage == RunningStage.PREDICTING:
            # Preprocess function for eval or predict
            tokenized_samples = self._prepare_val_features(samples, tokenized_samples)

            offset_mappings = tokenized_samples.pop("offset_mapping")
            example_ids = tokenized_samples.pop("example_id")
            contexts = tokenized_samples.pop("context")
            answers = tokenized_samples.pop("answer")

            tokenized_samples[DefaultDataKeys.METADATA] = []
            for offset_mapping, example_id, context in zip(offset_mappings, example_ids, contexts):
                tokenized_samples[DefaultDataKeys.METADATA].append(
                    {"context": context, "offset_mapping": offset_mapping, "example_id": example_id}
                )
            if self._running_stage.evaluating:
                for index, answer in enumerate(answers):
                    tokenized_samples[DefaultDataKeys.METADATA][index]["answer"] = answer

            del offset_mappings
            del example_ids
            del contexts
            del answers

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
        tokenized_samples["context"] = []
        tokenized_samples["answer"] = []

        for i in range(len(tokenized_samples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_samples.sequence_ids(i)
            context_index = 1 if self.pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_samples["example_id"].append(samples["id"][sample_index])
            tokenized_samples["context"].append(samples["context"][sample_index])
            if self._running_stage.evaluating:
                tokenized_samples["answer"].append(samples["answer"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_samples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_samples["offset_mapping"][i])
            ]

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


class QuestionAnsweringFileDataSource(QuestionAnsweringDataSource):
    def __init__(
        self,
        filetype: str,
        backbone: str,
        max_source_length: int = 384,
        max_target_length: int = 30,
        padding: Union[str, bool] = "max_length",
        question_column_name: Optional[str] = "question",
        context_column_name: Optional[str] = "context",
        answer_column_name: Optional[str] = "answer",
        doc_stride: Optional[int] = 128,
    ):
        super().__init__(
            backbone,
            max_source_length,
            max_target_length,
            padding,
            question_column_name,
            context_column_name,
            answer_column_name,
            doc_stride,
        )

        self.filetype = filetype

    def _reshape_answer_column(self, sample: Any):
        text = sample.pop("answer_text")
        start = sample.pop("answer_start")
        if isinstance(text, str):
            text = [text]
        if isinstance(start, int):
            start = [start]
        sample["answer"] = {"text": text, "answer_start": start}
        return sample

    def load_data(self, data: Any, columns: List[str] = None) -> "datasets.Dataset":
        if self.filetype == "json":
            file, field = data
        else:
            file = data

        data_files = {}
        stage = self._running_stage.value
        data_files[stage] = str(file)

        # FLASH_TESTING is set in the CI to run faster.
        if flash._IS_TESTING:
            try:
                if self.filetype == "json" and field is not None:
                    dataset_dict = DatasetDict(
                        {
                            stage: load_dataset(
                                self.filetype, data_files=data_files, split=[f"{stage}[:20]"], field=field
                            )[0]
                        }
                    )
                else:
                    dataset_dict = DatasetDict(
                        {stage: load_dataset(self.filetype, data_files=data_files, split=[f"{stage}[:20]"])[0]}
                    )
                column_names = dataset_dict[stage].column_names
            except Exception:
                if self.filetype == "json" and field is not None:
                    dataset_dict = load_dataset(self.filetype, data_files=data_files, field=field)
                else:
                    dataset_dict = load_dataset(self.filetype, data_files=data_files)
                column_names = dataset_dict[stage].column_names
        else:
            if self.filetype == "json" and field is not None:
                dataset_dict = load_dataset(self.filetype, data_files=data_files, field=field)
            else:
                dataset_dict = load_dataset(self.filetype, data_files=data_files)
            column_names = dataset_dict[stage].column_names

        if self.answer_column_name == "answer":
            if "answer" not in column_names:
                if "answer_text" in column_names and "answer_start" in column_names:
                    dataset_dict = dataset_dict.map(self._reshape_answer_column, batched=False)
                    column_names = dataset_dict[stage].column_names
                else:
                    raise KeyError(
                        """Dataset must contain either \"answer\" key as dict type or "answer_text" and "answer_start"
                        as string and integer types."""
                    )
        if not isinstance(dataset_dict[stage][self.answer_column_name][0], Dict):
            raise TypeError(
                f'{self.answer_column_name} column should be of type dict with keys "text" and "answer_start"'
            )

        dataset_dict = dataset_dict.map(self._tokenize_fn, batched=True, remove_columns=column_names)
        return dataset_dict[stage]

    def predict_load_data(self, data: Any) -> Union["datasets.Dataset", List[Dict[str, torch.Tensor]]]:
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
        max_source_length: int = 384,
        max_target_length: int = 30,
        padding: Union[str, bool] = "max_length",
        question_column_name: Optional[str] = "question",
        context_column_name: Optional[str] = "context",
        answer_column_name: Optional[str] = "answer",
        doc_stride: Optional[int] = 128,
    ):
        super().__init__(
            "csv",
            backbone,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
            question_column_name=question_column_name,
            context_column_name=context_column_name,
            answer_column_name=answer_column_name,
            doc_stride=doc_stride,
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
        max_source_length: int = 384,
        max_target_length: int = 30,
        padding: Union[str, bool] = "max_length",
        question_column_name: Optional[str] = "question",
        context_column_name: Optional[str] = "context",
        answer_column_name: Optional[str] = "answer",
        doc_stride: Optional[int] = 128,
    ):
        super().__init__(
            "json",
            backbone,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
            question_column_name=question_column_name,
            context_column_name=context_column_name,
            answer_column_name=answer_column_name,
            doc_stride=doc_stride,
        )

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("tokenizer")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True)


class QuestionAnsweringDictionaryDataSource(QuestionAnsweringDataSource):
    def load_data(self, data: Any, columns: List[str] = None) -> "datasets.Dataset":
        stage = self._running_stage.value

        dataset_dict = DatasetDict({stage: Dataset.from_dict(data)})

        column_names = dataset_dict[stage].column_names

        dataset_dict = dataset_dict.map(self._tokenize_fn, batched=True, remove_columns=column_names)

        return dataset_dict[stage]

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("tokenizer")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True)


class SQuADDataSource(QuestionAnsweringDataSource):
    def load_data(self, data: str, dataset: Optional[Any] = None) -> "datasets.Dataset":
        stage = self._running_stage.value

        file_path = data

        path = Path(file_path)
        with open(path, "rb") as f:
            squad_v_2_dict = json.load(f)

        ids = []
        titles = []
        contexts = []
        questions = []
        answers = []
        for topic in squad_v_2_dict["data"]:
            title = topic["title"]
            for comprehension in topic["paragraphs"]:
                context = comprehension["context"]
                for qa in comprehension["qas"]:
                    question = qa["question"]
                    id = qa["id"]

                    _answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                    _answers = [answer["text"] for answer in qa["answers"]]

                    ids.append(id)
                    titles.append(title)
                    contexts.append(context)
                    questions.append(question)
                    answers.append(dict(text=_answers, answer_start=_answer_starts))

        dataset_dict = DatasetDict(
            {
                stage: Dataset.from_dict(
                    {"id": ids, "title": titles, "context": contexts, "question": questions, "answer": answers}
                )
            }
        )

        column_names = dataset_dict[stage].column_names

        dataset_dict = dataset_dict.map(self._tokenize_fn, batched=True, remove_columns=column_names)

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
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        backbone: str = "distilbert-base-uncased",
        max_source_length: int = 384,
        max_target_length: int = 30,
        padding: Union[str, bool] = "max_length",
        question_column_name: Optional[str] = "question",
        context_column_name: Optional[str] = "context",
        answer_column_name: Optional[str] = "answer",
        doc_stride: Optional[int] = 128,
    ):
        self.backbone = backbone
        self.max_target_length = max_target_length
        self.max_source_length = max_source_length
        self.padding = padding
        self.question_column_name = question_column_name
        self.context_column_name = context_column_name
        self.answer_column_name = answer_column_name
        self.doc_stride = doc_stride

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
                    doc_stride=doc_stride,
                ),
                DefaultDataSources.JSON: QuestionAnsweringJSONDataSource(
                    self.backbone,
                    max_source_length=max_source_length,
                    max_target_length=max_target_length,
                    padding=padding,
                    question_column_name=question_column_name,
                    context_column_name=context_column_name,
                    answer_column_name=answer_column_name,
                    doc_stride=doc_stride,
                ),
                "dict": QuestionAnsweringDictionaryDataSource(
                    self.backbone,
                    max_source_length=max_source_length,
                    max_target_length=max_target_length,
                    padding=padding,
                    question_column_name=question_column_name,
                    context_column_name=context_column_name,
                    answer_column_name=answer_column_name,
                    doc_stride=doc_stride,
                ),
                "squad_v2": SQuADDataSource(
                    self.backbone,
                    max_source_length=max_source_length,
                    max_target_length=max_target_length,
                    padding=padding,
                    doc_stride=doc_stride,
                ),
            },
            default_data_source="dict",
        )

        self.set_state(QuestionAnsweringBackboneState(self.backbone))

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            **self.transforms,
            "backbone": self.backbone,
            "max_source_length": self.max_source_length,
            "max_target_length": self.max_target_length,
            "padding": self.padding,
            "question_column_name": self.question_column_name,
            "context_column_name": self.context_column_name,
            "answer_column_name": self.answer_column_name,
            "doc_stride": self.doc_stride,
        }

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool):
        return cls(**state_dict)

    def collate(self, samples: Any) -> Tensor:
        """Override to convert a set of samples to a batch."""
        if self.padding != "max_length":
            data_collator = DataCollatorWithPadding(AutoTokenizer.from_pretrained(self.backbone, use_fast=True))
            return data_collator(samples)
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

    def uncollate(self, predicted_sentences: collections.OrderedDict) -> Any:
        uncollated_predicted_sentences = []
        for key in predicted_sentences:
            uncollated_predicted_sentences.append({key: predicted_sentences[key]})
        return uncollated_predicted_sentences

    @staticmethod
    def per_sample_transform(sample: Any) -> Any:
        """Transforms to apply to a single sample after splitting up the batch.

        Can involve both CPU and Device transforms as this is not applied in separate workers.
        """
        return sample

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
    def from_squad_v2(
        cls,
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
        """Creates a :class:`~flash.text.question_answering.data.QuestionAnsweringData` object from the given data
        JSON files in the SQuAD2.0 format.

        Args:
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
                train_file="train.json",
                doc_stride=128,
            )
        """
        return cls.from_data_source(
            "squad_v2",
            train_file,
            val_file,
            test_file,
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

    @classmethod
    def from_json(
        cls,
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
    ) -> "DataModule":
        """Creates a :class:`~flash.text.question_answering.QuestionAnsweringData` object from the given JSON files
        using the :class:`~flash.text.question_answering.QuestionAnsweringDataSource`of name
        :attr:`~flash.core.data.data_source.DefaultDataSources.JSON` from the passed or constructed
        :class:`~flash.text.question_answering.QuestionAnsweringPreprocess`.

        Args:
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

        .. note:: The following keyword arguments can be passed through to the preprocess_kwargs

            - backbone: The HF model to be used for the task.
            - max_source_length: Max length of the sequence to be considered during tokenization.
            - max_target_length: Max length of each answer to be produced.
            - padding: Padding type during tokenization. Defaults to 'max_length'.
            - question_column_name: The key in the JSON file to recognize the question field. Defaults to"question".
            - context_column_name: The key in the JSON file to recognize the context field. Defaults to "context".
            - answer_column_name: The key in the JSON file to recognize the answer field. Defaults to "answer".
            - doc_stride: The stride amount to be taken when splitting up a long document into chunks.

        Returns:
            The constructed data module.

        Examples::

            data_module = QuestionAnsweringData.from_json(
                train_file="train_data.json",
                train_transform={
                    "to_tensor_transform": torch.as_tensor,
                },
                backbone="distilbert-base-uncased",
                max_source_length=384,
                max_target_length=30,
                padding='max_length',
                question_column_name="question",
                context_column_name="context",
                answer_column_name="answer",
                doc_stride=128
            )
        """
        return cls.from_data_source(
            DefaultDataSources.JSON,
            (train_file, field),
            (val_file, field),
            (test_file, field),
            (predict_file, field),
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
    ) -> "DataModule":
        """Creates a :class:`~flash.core.data.data_module.DataModule` object from the given CSV files using the
        :class:`~flash.core.data.data_source.DataSource`
        of name :attr:`~flash.core.data.data_source.DefaultDataSources.CSV`
        from the passed or constructed :class:`~flash.core.data.process.Preprocess`.

        Args:
            input_fields: The field or fields (columns) in the CSV file to use for the input.
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
            sampler: The ``sampler`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            preprocess_kwargs: Additional keyword arguments to use when constructing the preprocess. Will only be used
                if ``preprocess = None``.

        .. note:: The following keyword arguments can be passed through to the preprocess_kwargs

            - backbone: The HF model to be used for the task.
            - max_source_length: Max length of the sequence to be considered during tokenization.
            - max_target_length: Max length of each answer to be produced.
            - padding: Padding type during tokenization. Defaults to 'max_length'.
            - question_column_name: The key in the JSON file to recognize the question field. Defaults to"question".
            - context_column_name: The key in the JSON file to recognize the context field. Defaults to "context".
            - answer_column_name: The key in the JSON file to recognize the answer field. Defaults to "answer".
            - doc_stride: The stride amount to be taken when splitting up a long document into chunks.

        Returns:
            The constructed data module.

        Examples::

            data_module = QuestionAnsweringData.from_csv(
                "input",
                "target",
                train_file="train_data.csv",
                train_transform={
                    "to_tensor_transform": torch.as_tensor,
                },
                backbone="distilbert-base-uncased",
                max_source_length=384,
                max_target_length=30,
                padding='max_length',
                question_column_name="question",
                context_column_name="context",
                answer_column_name="answer",
                doc_stride=128
            )
        """
        return cls.from_data_source(
            DefaultDataSources.CSV,
            train_file,
            val_file,
            test_file,
            predict_file,
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
