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
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Mapping, Optional, Union

import torch
from datasets import Dataset, DatasetDict, load_dataset
from datasets.utils.download_manager import GenerateMode
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor
from transformers import AutoTokenizer, default_data_collator
from transformers.modeling_outputs import SequenceClassifierOutput

from flash.core.classification import ClassificationPostprocess
from flash.data.auto_dataset import AutoDataset
from flash.data.data_module import DataModule
from flash.data.data_pipeline import DataPipeline
from flash.data.process import Preprocess
from flash.data.utils import _contains_any_tensor


@dataclass(unsafe_hash=True, frozen=True)
class TextClfState:
    label_to_class_mapping: dict


class TextClassificationPreprocess(Preprocess):

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        input: str,
        max_length: int,
        filetype: str = 'csv',
        target: Optional[str] = None,
        label_to_class_mapping: Optional[dict] = None
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.input = input
        self.filetype = filetype
        self.max_length = max_length
        self.label_to_class_mapping = label_to_class_mapping
        self.target = target
        self._tokenize_fn = partial(
            self._tokenize_fn,
            tokenizer=self.tokenizer,
            input=self.input,
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )

    def per_batch_transform(self, batch: Any) -> Any:
        if "labels" not in batch:
            # todo: understand why an extra dimension has been added.
            if batch["input_ids"].dim() == 3:
                batch["input_ids"] = batch["input_ids"].squeeze(0)
        return batch

    @staticmethod
    def _tokenize_fn(ex, tokenizer=None, input: str = None, max_length: int = None, **kwargs) -> Callable:
        if isinstance(ex, dict):
            ex = ex[input]
        return tokenizer(ex, max_length=max_length, **kwargs)

    def collate(self, samples: Any) -> Tensor:
        """Override to convert a set of samples to a batch"""
        if isinstance(samples, dict):
            samples = [samples]
        return default_data_collator(samples)

    def _transform_label(self, ex):
        ex[self.target] = self.label_to_class_mapping[ex[self.target]]
        return ex

    def load_data(
        self,
        file: str,
        dataset: AutoDataset,
        columns: List[str] = ["input_ids", "attention_mask", "labels"],
        use_full: bool = True
    ):
        data_files = {}

        stage = dataset.running_stage.value
        data_files[stage] = str(file)

        if use_full and os.getenv("FLASH_TESTING", "0") == "0":
            dataset_dict = load_dataset(self.filetype, data_files=data_files)
        else:
            # used for debugging. Avoid processing the entire dataset   # noqa E265
            dataset_dict = DatasetDict({
                stage: load_dataset(self.filetype, data_files=data_files, split=[f'{stage}[:20]'])[0]
            })

        dataset_dict = dataset_dict.map(
            self._tokenize_fn,
            batched=True,
        )

        if self.label_to_class_mapping is None and self.training:
            # stage should always be train in that case. Not checking this,
            # since this is implicitly done by our dataflow.
            self.label_to_class_mapping = {
                v: k
                for k, v in enumerate(list(sorted(list(set(dataset_dict[stage][self.target])))))
            }

        # convert labels to ids
        if not self.predicting:
            dataset_dict = dataset_dict.map(self._transform_label)

        dataset_dict = dataset_dict.map(
            self._tokenize_fn,
            batched=True,
        )

        if not self.predicting and self.target != "labels":
            dataset_dict.rename_column_(self.target, "labels")

        dataset_dict.set_format("torch", columns=columns)

        if not self.predicting:
            dataset.num_classes = len(self.label_to_class_mapping)

        return dataset_dict[stage]

    def predict_load_data(self, sample: Any, dataset: AutoDataset):
        if isinstance(sample, str) and os.path.isfile(sample) and sample.endswith(".csv"):
            return self.load_data(sample, dataset, columns=["input_ids", "attention_mask"])
        else:
            if isinstance(sample, str):
                sample = [sample]

            if isinstance(sample, list) and all(isinstance(s, str) for s in sample):
                return [self._tokenize_fn(s) for s in sample]

            else:
                raise MisconfigurationException("Currently, we support only list of sentences")


class TextClassificationPostProcess(ClassificationPostprocess):

    def per_batch_transform(self, batch: Any) -> Any:
        if isinstance(batch, SequenceClassifierOutput):
            batch = batch.logits
        return super().per_batch_transform(batch)


class TextClassificationData(DataModule):
    """Data Module for text classification tasks"""
    preprocess_cls = TextClassificationPreprocess
    postprocess_cls = TextClassificationPostProcess
    _preprocess_state: Optional[TextClfState] = None
    target: Optional[str] = None

    __flash_special_attr__ = (
        "tokenizer", "input", "filetype", "target", "max_length", "_label_to_class_mapping", '_preprocess_state'
    )

    @property
    def preprocess_state(self) -> TextClfState:
        if self._preprocess_state is None or (
            self._label_to_class_mapping is not None
            and self._preprocess_state.label_to_class_mapping != self._label_to_class_mapping
        ):
            return TextClfState(self._label_to_class_mapping)

        return self._preprocess_state

    @preprocess_state.setter
    def preprocess_state(self, preprocess_state: TextClfState):
        self._preprocess_state = preprocess_state

    @property
    def label_to_class_mapping(self) -> Optional[Mapping]:
        mapping = self._label_to_class_mapping

        if mapping is None:
            if self._preprocess_state is not None:
                mapping = self._preprocess_state.label_to_class_mapping
            elif self.preprocess.label_to_class_mapping is not None:
                mapping = self.preprocess.label_to_class_mapping

        self._label_to_class_mapping = mapping

        return mapping

    @label_to_class_mapping.setter
    def label_to_class_mapping(self, new_mapping: Mapping):
        self._label_to_class_mapping = new_mapping

    @property
    def num_classes(self):
        if self._train_ds is not None and hasattr(self._train_ds, 'num_classes'):
            return self._train_ds.num_classes
        elif self._predict_ds is not None and hasattr(self._predict_ds, 'num_classes'):
            return self._predict_ds.num_classes
        return len(self.label_to_class_mapping)

    @property
    def preprocess(self) -> TextClassificationPreprocess:
        label_to_cls_mapping = self._label_to_class_mapping

        if label_to_cls_mapping is None and self.preprocess_state is not None:
            label_to_cls_mapping = self.preprocess_state.label_to_class_mapping
        return self.preprocess_cls(
            tokenizer=self.tokenizer,
            input=self.input,
            max_length=self.max_length,
            target=self.target,
            filetype=self.filetype,
            label_to_class_mapping=label_to_cls_mapping,
        )

    @classmethod
    def from_files(
        cls,
        train_file: Optional[str],
        input: str = 'input',
        target: Optional[str] = 'labels',
        filetype: str = "csv",
        backbone: str = "prajjwal1/bert-tiny",
        valid_file: Optional[str] = None,
        test_file: Optional[str] = None,
        predict_file: Optional[str] = None,
        max_length: int = 128,
        label_to_class_mapping: Optional[dict] = None,
        batch_size: int = 16,
        num_workers: Optional[int] = None,
    ) -> 'TextClassificationData':
        """Creates a TextClassificationData object from files.

        Args:
            train_file: Path to training data.
            input: The field storing the text to be classified.
            target: The field storing the class id of the associated text.
            filetype: .csv or .json
            backbone: tokenizer to use, can use any HuggingFace tokenizer.
            valid_file: Path to validation data.
            test_file: Path to test data.
            batch_size: the batchsize to use for parallel loading. Defaults to 64.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to None which equals the number of available CPU threads,
            or 0 for Darwin platform.

        Returns:
            TextClassificationData: The constructed data module.

        Examples::

            train_df = pd.read_csv("train_data.csv")
            tab_data = TabularData.from_df(train_df, target="fraud",
                                           numerical_input=["account_value"],
                                           categorical_input=["account_type"])

        """
        cls.tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
        cls.input = input
        cls.filetype = filetype
        cls.target = target
        cls.max_length = max_length
        cls._label_to_class_mapping = label_to_class_mapping

        return cls.from_load_data_inputs(
            train_load_data_input=train_file,
            valid_load_data_input=valid_file,
            test_load_data_input=test_file,
            predict_load_data_input=predict_file,
            batch_size=batch_size,
            num_workers=num_workers
        )

    @classmethod
    def from_file(
        cls,
        predict_file: str,
        input: str,
        backbone="bert-base-cased",
        filetype="csv",
        max_length: int = 128,
        preprocess_state: Optional[TextClfState] = None,
        label_to_class_mapping: Optional[dict] = None,
        batch_size: int = 16,
        num_workers: Optional[int] = None,
    ) -> 'TextClassificationData':
        """Creates a TextClassificationData object from files.

        Args:

            train_file: Path to training data.
            input: The field storing the text to be classified.
            filetype: .csv or .json
            backbone: tokenizer to use, can use any HuggingFace tokenizer.
            batch_size: the batchsize to use for parallel loading. Defaults to 64.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to None which equals the number of available CPU threads,
            or 0 for Darwin platform.
        """
        if preprocess_state is not None:
            cls._preprocess_state = preprocess_state

        return cls.from_files(
            None,
            input=input,
            target=None,
            filetype=filetype,
            backbone=backbone,
            valid_file=None,
            test_file=None,
            predict_file=predict_file,
            max_length=max_length,
            label_to_class_mapping=label_to_class_mapping,
            batch_size=batch_size,
            num_workers=num_workers,
        )
