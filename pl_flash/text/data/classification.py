import pathlib
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from datasets import ClassLabel, DatasetDict, Value, load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from pl_flash.data.datamodule import DataModule


class TextClassificationData(DataModule):
    """Data module for text classification tasks."""

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
        num_workers=None,
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
            tab_data = TabularData.from_df(train_df, target_col="fraud",
                                           numerical_cols=["account_value"],
                                           categorical_cols=["account_type"])

        """
        tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)

        paths = {"train": train_file}
        if valid_file is not None:
            paths["validation"] = valid_file
        if test_file is not None:
            paths["test"] = test_file

        dataset_dict = load_dataset(filetype, data_files=paths)

        label_to_class_mapping = {
            v: k for k, v in enumerate(list(sorted(list(set(dataset_dict["train"][label_field])))))
        }

        def transform_label(ex):
            ex[label_field] = label_to_class_mapping[ex[label_field]]
            return ex

        # convert labels to ids
        dataset_dict = dataset_dict.map(transform_label)

        # tokenize text field
        dataset_dict = dataset_dict.map(
            lambda ex: tokenizer(
                ex[text_field],
                max_length=max_length,
                truncation=True,
                padding="max_length",
            ),
            batched=True,
        )

        if label_field != "labels":
            dataset_dict.rename_column_(label_field, "labels")
        dataset_dict.set_format("torch", columns=["input_ids", "labels"])

        train_ds = dataset_dict["train"]
        valid_ds = None
        test_ds = None

        if "validation" in dataset_dict:
            valid_ds = dataset_dict["validation"]

        if "test" in dataset_dict:
            test_ds = dataset_dict["test"]

        datamodule = cls(
            train_ds=train_ds,
            valid_ds=valid_ds,
            test_ds=test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        datamodule.num_classes = len(label_to_class_mapping)
        return datamodule
