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
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from pytorch_lightning.trainer.states import RunningStage
from torch import Tensor

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
#   1) Decision related
#       i) Do we need a custom Deserializer ?
#       ii) How to route the {question, context, answer}_column_name values to the class constructor ?
#           1) Through the `load_data` method.
#           2) Keep it same and ask the user to model data according to the "question, context, answer" column names.
#   2)  Updates needed
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
        question_column_name: str = "question",
        context_column_name: str = "context",
        answer_column_name: str = "answers"
    ):
        super().__init__()

        self.backbone = backbone
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.padding = padding

        # Setup global pre-processing requirements
        #   pad_on_right, doc_stride, etc. are remaining
        self.question_column_name = question_column_name
        self.context_column_name = context_column_name
        self.answer_column_name = answer_column_name

    def _tokenize_fn(
        self,
        ex: Union[Dict[str, str], str],
        input: Optional[str] = None,
        target: Optional[str] = None,
    ) -> Callable:

        # Setup the tokenizing call here
        #   tokenized_samples = self.tokenizer(...params)
        tokenized_samples = self.tokenizer()

        stage = self._running_stage.value

        if stage == RunningStage.TRAINING:
            # Preprocess function for training
            self._prepare_train_features(tokenized_samples)
            pass

        if self._running_stage.evaluating() or stage == RunningStage.PREDICTING:
            # Preprocess function for eval or predict
            self._prepare_val_features(tokenized_samples)
            pass

        return tokenized_samples

    def _prepare_train_features(self, tokenized_samples: Any):
        # Example labelling step
        pass

    def _prepare_val_features(self, tokenized_samples: Any):
        # Relevant step
        pass

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("tokenizer")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True)


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
            file, input, target, field = data
        else:
            file, input, target = data
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

        file_path = data

        path = Path(file_path)
        with open(path, 'rb') as f:
            squad_v_2_dict = json.load(f)

        contexts = []
        questions = []
        answers = []
        for topic in squad_v_2_dict['data']:
            for comprehension in topic['paragraphs']:
                context = comprehension['context']
                for q_a_pair in comprehension['qas']:
                    question = q_a_pair['question']
                    for answer in q_a_pair['answers']:
                        answer_text = answer['text']

                        contexts.append(context)
                        questions.append(question)
                        answers.append(answer_text)

        dataset_dict = DatasetDict({
            stage: Dataset.from_dict({
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
        """Creates a :class:`~flash.text.question_answering.data.QuestionAnsweringData` object from the given
        data JSON files in the SQuAD2.0 format.

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
