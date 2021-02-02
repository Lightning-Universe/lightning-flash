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

from datasets import load_dataset
from torch import Tensor
from transformers import AutoTokenizer, default_data_collator

from flash.core.data import DataModule, TaskDataPipeline


def prepare_dataset(
    test_file: str,
    filetype: str,
    pipeline: TaskDataPipeline,
    train_file: Optional[str] = None,
    valid_file: Optional[str] = None,
    predict: bool = False
):
    data_files = {}

    if train_file is not None:
        data_files["train"] = train_file
    if valid_file is not None:
        data_files["validation"] = valid_file
    if test_file is not None:
        data_files["test"] = test_file

    # load the dataset
    dataset_dict = load_dataset(
        filetype,
        data_files=data_files,
    )

    # tokenize the dataset
    dataset_dict = dataset_dict.map(
        pipeline._tokenize_fn,
        batched=True,
    )
    columns = ["input_ids", "attention_mask"] if predict else ["input_ids", "attention_mask", "labels"]
    dataset_dict.set_format(columns=columns)

    train_ds = None
    valid_ds = None
    test_ds = None

    if "train" in dataset_dict:
        train_ds = dataset_dict["train"]

    if "validation" in dataset_dict:
        valid_ds = dataset_dict["validation"]

    if "test" in dataset_dict:
        test_ds = dataset_dict["test"]

    return train_ds, valid_ds, test_ds


class Seq2SeqDataPipeline(TaskDataPipeline):

    def __init__(
        self,
        tokenizer,
        input: str,
        target: Optional[str] = None,
        max_source_length: int = 128,
        max_target_length: int = 128,
        padding: Union[str, bool] = 'longest'
    ):
        self.tokenizer = tokenizer
        self._input = input
        self._target = target
        self._max_target_length = max_target_length
        self._max_source_length = max_source_length
        self._padding = padding
        self._tokenize_fn = partial(
            self._tokenize_fn,
            tokenizer=self.tokenizer,
            input=self._input,
            target=self._target,
            max_source_length=self._max_source_length,
            max_target_length=self._max_target_length,
            padding=self._padding
        )

    def before_collate(self, samples: Any) -> Any:
        """Override to apply transformations to samples"""
        if isinstance(samples, (list, tuple)) and len(samples) > 0 and all(isinstance(s, str) for s in samples):
            return [self._tokenize_fn({self._input: s, self._target: None}) for s in samples]
        return samples

    @staticmethod
    def _tokenize_fn(
        ex,
        tokenizer,
        input: str,
        target: Optional[str],
        max_source_length: int,
        max_target_length: int,
        padding: Union[str, bool],
    ) -> Callable:
        output = tokenizer.prepare_seq2seq_batch(
            src_texts=ex[input],
            tgt_texts=ex[target] if target else None,
            max_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
        )
        return output

    def collate(self, samples: Any) -> Tensor:
        """Override to convert a set of samples to a batch"""
        return default_data_collator(samples)

    def after_collate(self, batch: Any) -> Any:
        return batch

    def uncollate(self, generated_tokens: Any) -> Any:
        pred_str = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        pred_str = [str.strip(s) for s in pred_str]
        return pred_str


class Seq2SeqData(DataModule):
    """Data module for Seq2Seq tasks."""

    @staticmethod
    def default_pipeline():
        return Seq2SeqDataPipeline(
            AutoTokenizer.from_pretrained("sshleifer/tiny-mbart", use_fast=True),
            input="input",
        )

    @classmethod
    def from_files(
        cls,
        train_file: str,
        input: str = 'input',
        target: Optional[str] = None,
        filetype: str = "csv",
        backbone: str = "sshleifer/tiny-mbart",
        valid_file: Optional[str] = None,
        test_file: Optional[str] = None,
        max_source_length: int = 128,
        max_target_length: int = 128,
        padding: Union[str, bool] = 'max_length',
        batch_size: int = 32,
        num_workers: Optional[int] = None,
    ):
        """Creates a Seq2SeqData object from files.

        Args:
            train_file: Path to training data.
            input: The field storing the source translation text.
            target: The field storing the target translation text.
            filetype: .csv or .json
            backbone: tokenizer to use, can use any HuggingFace tokenizer.
            valid_file: Path to validation data.
            test_file: Path to test data.
            max_source_length: Maximum length of the source text. Any text longer will be truncated.
            max_target_length: Maximum length of the target text. Any text longer will be truncated.
            padding: Padding strategy for batches. Default is pad to maximum length.
            batch_size: the batchsize to use for parallel loading. Defaults to 32.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to None which equals the number of available CPU threads.

        Returns:
            Seq2SeqData: The constructed data module.

        Examples::

            train_df = pd.read_csv("train_data.csv")
            tab_data = TabularData.from_df(train_df, target="fraud",
                                           numerical_input=["account_value"],
                                           categorical_input=["account_type"])

        """
        tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)

        pipeline = Seq2SeqDataPipeline(
            tokenizer=tokenizer,
            input=input,
            target=target,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding
        )

        train_ds, valid_ds, test_ds = prepare_dataset(
            train_file=train_file, valid_file=valid_file, test_file=test_file, filetype=filetype, pipeline=pipeline
        )

        datamodule = cls(
            train_ds=train_ds,
            valid_ds=valid_ds,
            test_ds=test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        datamodule.data_pipeline = pipeline
        return datamodule

    @classmethod
    def from_file(
        cls,
        predict_file: str,
        input: str = 'input',
        target: Optional[str] = None,
        backbone: str = "sshleifer/tiny-mbart",
        filetype: str = "csv",
        max_source_length: int = 128,
        max_target_length: int = 128,
        padding: Union[str, bool] = 'max_length',
        batch_size: int = 32,
        num_workers: Optional[int] = None,
    ):
        """Creates a TextClassificationData object from files.

        Args:
            predict_file: Path to prediction input file.
            input: The field storing the source translation text.
            target: The field storing the target translation text.
            backbone: tokenizer to use, can use any HuggingFace tokenizer.
            filetype: csv or json.
            max_source_length: Maximum length of the source text. Any text longer will be truncated.
            max_target_length: Maximum length of the target text. Any text longer will be truncated.
            padding: Padding strategy for batches. Default is pad to maximum length.
            batch_size: the batchsize to use for parallel loading. Defaults to 32.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to None which equals the number of available CPU threads.

        Returns:
            Seq2SeqData: The constructed data module.

        """
        tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)

        pipeline = Seq2SeqDataPipeline(
            tokenizer=tokenizer,
            input=input,
            target=target,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding
        )

        train_ds, valid_ds, test_ds = prepare_dataset(
            test_file=predict_file, filetype=filetype, pipeline=pipeline, predict=True
        )

        datamodule = cls(
            train_ds=train_ds,
            valid_ds=valid_ds,
            test_ds=test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        datamodule.data_pipeline = pipeline
        return datamodule
