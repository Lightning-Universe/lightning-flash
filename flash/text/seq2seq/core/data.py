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
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

import datasets
import torch
from datasets import DatasetDict, load_dataset
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor
from transformers import AutoTokenizer, default_data_collator

from flash.data.data_module import DataModule
from flash.data.process import Postprocess, Preprocess


class Seq2SeqPreprocess(Preprocess):

    def __init__(
        self,
        tokenizer,
        input: str,
        filetype: str,
        target: Optional[str] = None,
        max_source_length: int = 128,
        max_target_length: int = 128,
        padding: Union[str, bool] = 'longest'
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.input = input
        self.filetype = filetype
        self.target = target
        self.max_target_length = max_target_length
        self.max_source_length = max_source_length
        self.padding = padding
        self._tokenize_fn = partial(
            self._tokenize_fn,
            tokenizer=self.tokenizer,
            input=self.input,
            target=self.target,
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
            padding=self.padding
        )

    @staticmethod
    def version() -> str:
        return "0.0.1"

    def save_state_dict(self) -> Dict[str, Any]:
        return {
            "input": self.input,
            "backbone": self.backbone,
            "max_length": self.max_length,
            "target": self.target,
            "filetype": self.filetype,
            "label_to_class_mapping": self.label_to_class_mapping,
        }

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], keep_vars: bool):
        return cls(**state_dict)

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

    def load_data(
        self,
        file: str,
        use_full: bool = True,
        columns: List[str] = ["input_ids", "attention_mask", "labels"]
    ) -> 'datasets.Dataset':
        data_files = {}
        stage = self._running_stage.value
        data_files[stage] = str(file)

        # FLASH_TESTING is set in the CI to run faster.
        if use_full and os.getenv("FLASH_TESTING", "0") == "0":
            dataset_dict = load_dataset(self.filetype, data_files=data_files)
        else:
            # used for debugging. Avoid processing the entire dataset   # noqa E265
            try:
                dataset_dict = DatasetDict({
                    stage: load_dataset(self.filetype, data_files=data_files, split=[f'{stage}[:20]'])[0]
                })
            except AssertionError:
                dataset_dict = load_dataset(self.filetype, data_files=data_files)

        dataset_dict = dataset_dict.map(self._tokenize_fn, batched=True)
        dataset_dict.set_format(columns=columns)
        return dataset_dict[stage]

    def predict_load_data(self, sample: Any) -> Union['datasets.Dataset', List[Dict[str, torch.Tensor]]]:
        if isinstance(sample, str) and os.path.isfile(sample) and sample.endswith(".csv"):
            return self.load_data(sample, use_full=True, columns=["input_ids", "attention_mask"])
        else:
            if isinstance(sample, (list, tuple)) and len(sample) > 0 and all(isinstance(s, str) for s in sample):
                return [self._tokenize_fn({self.input: s, self.target: None}) for s in sample]
            else:
                raise MisconfigurationException("Currently, we support only list of sentences")

    def collate(self, samples: Any) -> Tensor:
        """Override to convert a set of samples to a batch"""
        return default_data_collator(samples)


class Seq2SeqData(DataModule):
    """Data module for Seq2Seq tasks."""

    preprocess_cls = Seq2SeqPreprocess

    @classmethod
    def from_files(
        cls,
        train_file: Optional[str],
        input: str = 'input',
        target: Optional[str] = None,
        filetype: str = "csv",
        backbone: str = "sshleifer/tiny-mbart",
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        predict_file: Optional[str] = None,
        max_source_length: int = 128,
        max_target_length: int = 128,
        padding: Union[str, bool] = 'max_length',
        batch_size: int = 32,
        num_workers: Optional[int] = None,
        preprocess: Optional[Preprocess] = None,
        postprocess: Optional[Postprocess] = None,
    ):
        """Creates a Seq2SeqData object from files.
        Args:
            train_file: Path to training data.
            input: The field storing the source translation text.
            target: The field storing the target translation text.
            filetype: ``csv`` or ``json`` File
            backbone: Tokenizer backbone to use, can use any HuggingFace tokenizer.
            val_file: Path to validation data.
            test_file: Path to test data.
            max_source_length: Maximum length of the source text. Any text longer will be truncated.
            max_target_length: Maximum length of the target text. Any text longer will be truncated.
            padding: Padding strategy for batches. Default is pad to maximum length.
            batch_size: The batchsize to use for parallel loading. Defaults to 32.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to None which equals the number of available CPU threads,
                or 0 for Darwin platform.
        Returns:
            Seq2SeqData: The constructed data module.
        Examples::
            train_df = pd.read_csv("train_data.csv")
            tab_data = TabularData.from_df(train_df,
                                           target="fraud",
                                           num_cols=["account_value"],
                                           cat_cols=["account_type"])
        """
        tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
        preprocess = preprocess or cls.preprocess_cls(
            tokenizer,
            input,
            filetype,
            target,
            max_source_length,
            max_target_length,
            padding,
        )

        return cls.from_load_data_inputs(
            train_load_data_input=train_file,
            val_load_data_input=val_file,
            test_load_data_input=test_file,
            predict_load_data_input=predict_file,
            batch_size=batch_size,
            num_workers=num_workers,
            preprocess=preprocess,
            postprocess=postprocess,
        )

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
        preprocess: Optional[Preprocess] = None,
        postprocess: Optional[Postprocess] = None,
    ):
        """Creates a TextClassificationData object from files.
        Args:
            predict_file: Path to prediction input file.
            input: The field storing the source translation text.
            target: The field storing the target translation text.
            backbone: Tokenizer backbone to use, can use any HuggingFace tokenizer.
            filetype: Csv or json.
            max_source_length: Maximum length of the source text. Any text longer will be truncated.
            max_target_length: Maximum length of the target text. Any text longer will be truncated.
            padding: Padding strategy for batches. Default is pad to maximum length.
            batch_size: The batchsize to use for parallel loading. Defaults to 32.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to None which equals the number of available CPU threads,
                or 0 for Darwin platform.
        Returns:
            Seq2SeqData: The constructed data module.
        """
        return cls.from_files(
            train_file=None,
            input=input,
            target=target,
            filetype=filetype,
            backbone=backbone,
            predict_file=predict_file,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
            batch_size=batch_size,
            num_workers=num_workers,
            preprocess=preprocess,
            postprocess=postprocess,
        )
