from typing import Any, Callable

import torch
from datasets import load_dataset
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor
from transformers import AutoTokenizer, default_data_collator

from flash.core.classification import ClassificationDataPipeline
from flash.core.data import DataModule


def tokenize_text_lambda(tokenizer, text_field, max_length):
    return lambda ex: tokenizer(
        ex[text_field],
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
    text_field,
    max_length,
    label_field=None,
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

    dataset_dict = load_dataset(filetype, data_files=data_files)

    if not predict:
        if label_to_class_mapping is None:
            label_to_class_mapping = {
                v: k
                for k, v in enumerate(list(sorted(list(set(dataset_dict["train"][label_field])))))
            }

        def transform_label(ex):
            ex[label_field] = label_to_class_mapping[ex[label_field]]
            return ex

            # convert labels to ids

        dataset_dict = dataset_dict.map(transform_label)

    # tokenize text field
    dataset_dict = dataset_dict.map(
        tokenize_text_lambda(tokenizer, text_field, max_length),
        batched=True,
    )

    if label_field != "labels" and not predict:
        dataset_dict.rename_column_(label_field, "labels")
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

    def __init__(self, tokenizer, text_field: str, max_length: int):
        self._tokenizer = tokenizer
        self._text_field = text_field
        self._max_length = max_length
        self._tokenize_fn = self.tokenize_text_lambda(self._tokenizer, self._text_field, self._max_length)

    @staticmethod
    def tokenize_text_lambda(tokenizer, text_field: str, max_length: str) -> Callable:
        return lambda ex: tokenizer(
            ex[text_field],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

    def before_collate(self, samples: Any) -> Any:
        """Override to apply transformations to samples"""
        if self.contains_any_tensor(samples):
            return samples

        elif isinstance(samples, (list, tuple)) and len(samples) > 0 and all(isinstance(s, str) for s in samples):
            return [self._tokenize_fn({self._text_field: s}) for s in samples]

        else:
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
        else:
            return batch

    def before_uncollate(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.softmax(batch[0], -1)


class TextClassificationData(DataModule):
    """Data module for text classification tasks."""

    @staticmethod
    def default_pipeline():
        return TextClassificationDataPipeline(
            AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True),
            "sentiment",  # Todo: find a way to get the target column name or impose target
            128,
        )

    @classmethod
    def from_files(
        cls,
        train_file,
        text_field,
        label_field,
        filetype="csv",
        backbone="bert-base-cased",
        valid_file=None,
        test_file=None,
        max_length: int = 128,
        batch_size=1,
        num_workers=0,
    ):
        """Creates a TextClassificationData object from files.

        Args:
            train_file: Path to training data.
            text_field: The field storing the text to be classified.
            label_field: The field storing the class id of the associated text.
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
            text_field,
            max_length,
            label_field=label_field,
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
        datamodule.data_pipeline = TextClassificationDataPipeline(
            tokenizer, text_field=text_field, max_length=max_length
        )
        return datamodule

    @classmethod
    def from_file(
        cls,
        predict_file,
        text_field,
        backbone="bert-base-cased",
        filetype="csv",
        max_length: int = 128,
        batch_size=2,
        num_workers=0,
    ):
        """Creates a TextClassificationData object from files.

        Args:
            train_file: Path to training data.
            text_field: The field storing the text to be classified.
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
            text_field,
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

        datamodule.data_pipeline = TextClassificationDataPipeline(
            tokenizer, text_field=text_field, max_length=max_length
        )
        return datamodule
