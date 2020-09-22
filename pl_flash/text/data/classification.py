import pathlib
from typing import Sequence, Callable, Optional, Union, Any, Tuple

import torch
from pl_flash import DataModule

from transformers import AutoTokenizer
from datasets import DatasetDict, load_dataset, ClassLabel, Value


class TextClassificationData(DataModule):
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
        tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)

        paths = {"train": train_file}
        if valid_file is not None:
            paths["validation"] = valid_file
        if test_file is not None:
            paths["test"] = test_file

        dataset_dict = load_dataset(filetype, data_files=paths)

        # tokenize text field
        dataset_dict = dataset_dict.map(
            lambda ex: tokenizer(
                ex[text_field],
                max_length=max_length,
                truncation=True,
            ),
            batched=True,
        )

        if label_field != "labels":
            dataset_dict.rename_column_(label_field, "labels")
        dataset_dict.set_format("torch", columns=["input_ids", "labels"])

        return cls(
            train_ds=dataset_dict["train"],
            valid_ds=dataset_dict.get("validation", None),
            test_ds=dataset_dict.get("test", None),
            batch_size=batch_size,
            num_workers=num_workers,
        )
