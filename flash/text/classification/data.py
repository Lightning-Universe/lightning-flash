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
from functools import partial
from typing import Any, Callable, Optional, Union

import torch
from datasets import load_dataset
from datasets.utils.download_manager import GenerateMode
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor
from transformers import AutoTokenizer, default_data_collator
from transformers.modeling_outputs import SequenceClassifierOutput

from flash.core.classification import ClassificationDataPipeline
from flash.core.data import DataModule
from flash.core.data.utils import _contains_any_tensor


def tokenize_text_lambda(tokenizer, input, max_length):
    return lambda ex: tokenizer(
        ex[input],
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )


def prepare_dataset(
    tokenizer,
    train_file,
    valid_file,
    test_file,
    filetype,
    backbone,
    input,
    max_length,
    target=None,
    label_to_class_mapping=None,
    predict=False,
):
    data_files = {}

    if train_file is not None:
        data_files["train"] = train_file
    if valid_file is not None:
        data_files["validation"] = valid_file
    if test_file is not None:
        data_files["test"] = test_file

    dataset_dict = load_dataset(filetype, data_files=data_files, download_mode=GenerateMode.FORCE_REDOWNLOAD)

    if not predict:
        if label_to_class_mapping is None:
            label_to_class_mapping = {
                v: k
                for k, v in enumerate(list(sorted(list(set(dataset_dict["train"][target])))))
            }

        def transform_label(ex):
            ex[target] = label_to_class_mapping[ex[target]]
            return ex

            # convert labels to ids

        dataset_dict = dataset_dict.map(transform_label)

    # tokenize text field
    dataset_dict = dataset_dict.map(
        tokenize_text_lambda(tokenizer, input, max_length),
        batched=True,
    )

    if target != "labels" and not predict:
        dataset_dict.rename_column_(target, "labels")
    dataset_dict.set_format("torch", columns=["input_ids"] if predict else ["input_ids", "labels"])

    train_ds = None
    valid_ds = None
    test_ds = None

    if "train" in dataset_dict:
        train_ds = dataset_dict["train"]

    if "validation" in dataset_dict:
        valid_ds = dataset_dict["validation"]

    if "test" in dataset_dict:
        test_ds = dataset_dict["test"]

    return train_ds, valid_ds, test_ds, label_to_class_mapping


class TextClassificationDataPipeline(ClassificationDataPipeline):

    def __init__(self, tokenizer, input: str, max_length: int):
        self._tokenizer = tokenizer
        self._input = input
        self._max_length = max_length
        self._tokenize_fn = partial(
            self._tokenize_fn, tokenizer=self._tokenizer, input=self._input, max_length=self._max_length
        )

    @staticmethod
    def _tokenize_fn(ex, tokenizer=None, input: str = None, max_length: int = None) -> Callable:
        return tokenizer(
            ex[input],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

    def before_collate(self, samples: Any) -> Any:
        """Override to apply transformations to samples"""
        if _contains_any_tensor(samples):
            return samples
        elif isinstance(samples, (list, tuple)) and len(samples) > 0 and all(isinstance(s, str) for s in samples):
            return [self._tokenize_fn({self._input: s}) for s in samples]
        raise MisconfigurationException("samples can only be tensors or a list of sentences.")

    def collate(self, samples: Any) -> Tensor:
        """Override to convert a set of samples to a batch"""
        if isinstance(samples, dict):
            samples = [samples]
        return default_data_collator(samples)

    def after_collate(self, batch: Tensor) -> Tensor:
        if "labels" not in batch:
            # todo: understand why an extra dimension has been added.
            if batch["input_ids"].dim() == 3:
                batch["input_ids"] = batch["input_ids"].squeeze(0)
        return batch

    def before_uncollate(self, batch: Union[torch.Tensor, tuple,
                                            SequenceClassifierOutput]) -> Union[tuple, torch.Tensor]:
        if isinstance(batch, SequenceClassifierOutput):
            batch = batch.logits
        return super().before_uncollate(batch)


class TextClassificationData(DataModule):
    """Data module for text classification tasks."""

    @staticmethod
    def default_pipeline():
        return TextClassificationDataPipeline(
            AutoTokenizer.from_pretrained("prajjwal1/bert-tiny", use_fast=True),
            "sentiment",  # Todo: find a way to get the target column name or impose target
            128,
        )

    @classmethod
    def from_files(
        cls,
        train_file,
        input,
        target,
        filetype="csv",
        backbone="prajjwal1/bert-tiny",
        valid_file=None,
        test_file=None,
        max_length: int = 128,
        batch_size: int = 16,
        num_workers: Optional[int] = None,
    ):
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
                Defaults to None which equals the number of available CPU threads.

        Returns:
            TextClassificationData: The constructed data module.

        Examples::

            train_df = pd.read_csv("train_data.csv")
            tab_data = TabularData.from_df(train_df, target="fraud",
                                           numerical_input=["account_value"],
                                           categorical_input=["account_type"])

        """
        tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)

        train_ds, valid_ds, test_ds, label_to_class_mapping = prepare_dataset(
            tokenizer,
            train_file,
            valid_file,
            test_file,
            filetype,
            backbone,
            input,
            max_length,
            target=target,
            label_to_class_mapping=None
        )

        datamodule = cls(
            train_ds=train_ds,
            valid_ds=valid_ds,
            test_ds=test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        datamodule.num_classes = len(label_to_class_mapping)
        datamodule.data_pipeline = TextClassificationDataPipeline(tokenizer, input=input, max_length=max_length)
        return datamodule

    @classmethod
    def from_file(
        cls,
        predict_file: str,
        input: str,
        backbone="bert-base-cased",
        filetype="csv",
        max_length: int = 128,
        batch_size: int = 16,
        num_workers: Optional[int] = None,
    ):
        """Creates a TextClassificationData object from files.

        Args:
            train_file: Path to training data.
            input: The field storing the text to be classified.
            filetype: .csv or .json
            backbone: tokenizer to use, can use any HuggingFace tokenizer.
            batch_size: the batchsize to use for parallel loading. Defaults to 64.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to None which equals the number of available CPU threads.

        Returns:
            TextClassificationData: The constructed data module.

        """
        tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)

        _, _, predict_ds, _ = prepare_dataset(
            tokenizer,
            None,
            None,
            predict_file,
            filetype,
            backbone,
            input,
            max_length,
            predict=True,
        )

        datamodule = cls(
            train_ds=None,
            valid_ds=None,
            test_ds=predict_ds,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        datamodule.data_pipeline = TextClassificationDataPipeline(tokenizer, input=input, max_length=max_length)
        return datamodule
