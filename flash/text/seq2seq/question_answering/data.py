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
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from torch import Tensor

from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_source import DefaultDataSources
from flash.core.data.process import Preprocess
from flash.core.utilities.imports import _TEXT_AVAILABLE, requires_extras
from flash.text.classification.data import TextDeserializer
from flash.text.seq2seq.core.data import (
    Seq2SeqBackboneState,
    Seq2SeqCSVDataSource,
    Seq2SeqData,
    Seq2SeqDictionaryDataSource,
    Seq2SeqJSONDataSource,
    Seq2SeqPostprocess,
    Seq2SeqSentencesDataSource,
)

if _TEXT_AVAILABLE:
    import datasets
    from datasets import Dataset, DatasetDict
    from transformers import default_data_collator


class SQuADDataSource(Seq2SeqDictionaryDataSource):

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

        dataset_dict = dataset_dict.map(
            partial(self._tokenize_fn, input="question", input_pair="context", target="answer"), batched=True
        )

        return dataset_dict[stage]


class QuestionAnsweringPreprocess(Preprocess):

    @requires_extras("text")
    def __init__(
        self,
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
                DefaultDataSources.CSV: Seq2SeqCSVDataSource(
                    self.backbone,
                    max_source_length=max_source_length,
                    max_target_length=max_target_length,
                    padding=padding,
                ),
                DefaultDataSources.JSON: Seq2SeqJSONDataSource(
                    self.backbone,
                    max_source_length=max_source_length,
                    max_target_length=max_target_length,
                    padding=padding,
                ),
                "sentences": Seq2SeqSentencesDataSource(
                    self.backbone,
                    max_source_length=max_source_length,
                    max_target_length=max_target_length,
                    padding=padding,
                ),
                "dict": Seq2SeqDictionaryDataSource(
                    self.backbone,
                    max_source_length=max_source_length,
                    max_target_length=max_target_length,
                    padding=padding,
                ),
                "squad_v2": SQuADDataSource(
                    self.backbone,
                    max_source_length=max_source_length,
                    max_target_length=max_target_length,
                    padding=padding,
                )
            },
            # TODO: Change default here to Dictionary
            default_data_source="dict",
            deserializer=TextDeserializer(backbone, max_source_length)
        )

        self.set_state(Seq2SeqBackboneState(self.backbone))

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


class QuestionAnsweringData(Seq2SeqData):

    preprocess_cls = QuestionAnsweringPreprocess
    postprocess_cls = Seq2SeqPostprocess

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
        """Creates a :class:`~flash.image.detection.data.ObjectDetectionData` object from the given data
        folders and corresponding target folders.

        Args:
            train_folder: The folder containing the train data.
            train_ann_file: The COCO format annotation file.
            val_folder: The folder containing the validation data.
            val_ann_file: The COCO format annotation file.
            test_folder: The folder containing the test data.
            test_ann_file: The COCO format annotation file.
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

            data_module = SemanticSegmentationData.from_coco(
                train_folder="train_folder",
                train_ann_file="annotations.json",
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
