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
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, Union

from datasets import DatasetDict, load_dataset
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor
from transformers import AutoTokenizer, default_data_collator
from transformers.modeling_outputs import SequenceClassifierOutput

from flash.core.classification import ClassificationPostprocess
from flash.data.auto_dataset import AutoDataset
from flash.data.data_module import DataModule
from flash.data.process import Preprocess, PreprocessState


@dataclass(unsafe_hash=True, frozen=True)
class TextClassificationState(PreprocessState):
    label_to_class_mapping: Dict[str, int]


class TextClassificationPreprocess(Preprocess):

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        input: str,
        max_length: int,
        target: str,
        filetype: str,
        label_to_class_mapping: Dict[str, int],
    ):
        """
        This class contains the preprocessing logic for text classification

        Args:
            tokenizer: Hugging Face Tokenizer.
            input: The field storing the text to be classified.
            max_length:  Maximum number of tokens within a single sentence.
            target: The field storing the class id of the associated text.
            filetype: .csv or .json format type.
            label_to_class_mapping: Dictionnary mapping target labels to class indexes.

        Returns:
            TextClassificationPreprocess: The constructed preprocess objects.

        """

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

    @property
    def state(self):
        return TextClassificationState(self.label_to_class_mapping)

    def per_batch_transform(self, batch: Any) -> Any:
        if "labels" not in batch:
            # todo: understand why an extra dimension has been added.
            if batch["input_ids"].dim() == 3:
                batch["input_ids"] = batch["input_ids"].squeeze(0)
        return batch

    @staticmethod
    def _tokenize_fn(
        ex: Union[Dict[str, str], str],
        tokenizer=None,
        input: str = None,
        max_length: int = None,
        **kwargs
    ) -> Callable:
        """This function is used to tokenize sentences using the provided tokenizer."""
        if isinstance(ex, dict):
            ex = ex[input]
        return tokenizer(ex, max_length=max_length, **kwargs)

    def collate(self, samples: Any) -> Tensor:
        """Override to convert a set of samples to a batch"""
        if isinstance(samples, dict):
            samples = [samples]
        return default_data_collator(samples)

    def _transform_label(self, ex: Dict[str, str]):
        ex[self.target] = self.label_to_class_mapping[ex[self.target]]
        return ex

    @staticmethod
    def generate_state(file: str, target: str, filetype: str) -> TextClassificationState:
        data_files = {'train': file}
        dataset_dict = load_dataset(filetype, data_files=data_files)
        label_to_class_mapping = {v: k for k, v in enumerate(list(sorted(list(set(dataset_dict['train'][target])))))}
        return TextClassificationState(label_to_class_mapping)

    def load_data(
        self,
        filepath: str,
        dataset: AutoDataset,
        columns: Union[List[str], Tuple[str]] = ("input_ids", "attention_mask", "labels"),
        use_full: bool = True
    ):
        data_files = {}

        stage = dataset.running_stage.value
        data_files[stage] = str(filepath)

        # FLASH_TESTING is set in the CI to run faster.
        if use_full and os.getenv("FLASH_TESTING", "0") == "0":
            dataset_dict = load_dataset(self.filetype, data_files=data_files)
        else:
            # used for debugging. Avoid processing the entire dataset   # noqa E265
            dataset_dict = DatasetDict({
                stage: load_dataset(self.filetype, data_files=data_files, split=[f'{stage}[:20]'])[0]
            })

        dataset_dict = dataset_dict.map(self._tokenize_fn, batched=True)

        # convert labels to ids
        if not self.predicting:
            dataset_dict = dataset_dict.map(self._transform_label)

        dataset_dict = dataset_dict.map(self._tokenize_fn, batched=True)

        # Hugging Face models expect target to be named ``labels``.
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
    target: Optional[str] = None

    @property
    def preprocess_state(self) -> TextClassificationState:
        return self._preprocess.state

    @property
    def num_classes(self) -> int:
        return len(self.preprocess_state.label_to_class_mapping)

    @classmethod
    def instantiate_preprocess(
        cls,
        train_file: Optional[str],
        input: str,
        target: str,
        filetype: str,
        backbone: str,
        max_length: int,
        label_to_class_mapping: Optional[dict] = None,
        preprocess_state: Optional[TextClassificationState] = None,
        preprocess_cls: Optional[Type[Preprocess]] = None,
    ):
        if label_to_class_mapping is None:
            preprocess_cls = preprocess_cls or cls.preprocess_cls
            if train_file is not None:
                preprocess_state = preprocess_cls.generate_state(train_file, target, filetype)
            else:
                if preprocess_state is None:
                    raise MisconfigurationException(
                        "Either ``preprocess_state`` or ``train_file`` needs to be provided"
                    )
            label_to_class_mapping = preprocess_state.label_to_class_mapping

        preprocess_cls = preprocess_cls or cls.preprocess_cls

        return preprocess_cls(
            AutoTokenizer.from_pretrained(backbone, use_fast=True),
            input,
            max_length,
            target,
            filetype,
            label_to_class_mapping,
        )

    @classmethod
    def from_files(
        cls,
        train_file: Optional[str],
        input: Optional[str] = 'input',
        target: Optional[str] = 'labels',
        filetype: str = "csv",
        backbone: str = "prajjwal1/bert-tiny",
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        predict_file: Optional[str] = None,
        max_length: int = 128,
        label_to_class_mapping: Optional[dict] = None,
        batch_size: int = 16,
        num_workers: Optional[int] = None,
        preprocess_state: Optional[TextClassificationState] = None,
        preprocess_cls: Optional[Type[Preprocess]] = None,
    ) -> 'TextClassificationData':
        """Creates a TextClassificationData object from files.

        Args:
            train_file: Path to training data.
            input: The field storing the text to be classified.
            target: The field storing the class id of the associated text.
            filetype: .csv or .json
            backbone: Tokenizer backbone to use, can use any HuggingFace tokenizer.
            val_file: Path to validation data.
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
                                           num_cols=["account_value"],
                                           cat_cols=["account_type"])

        """
        preprocess = cls.instantiate_preprocess(
            train_file,
            input,
            target,
            filetype,
            backbone,
            max_length,
            label_to_class_mapping,
            preprocess_state,
            preprocess_cls,
        )

        return cls.from_load_data_inputs(
            train_load_data_input=train_file,
            val_load_data_input=val_file,
            test_load_data_input=test_file,
            predict_load_data_input=predict_file,
            batch_size=batch_size,
            num_workers=num_workers,
            preprocess=preprocess
        )

    @classmethod
    def from_file(
        cls,
        predict_file: str,
        input: str,
        backbone="bert-base-cased",
        filetype="csv",
        max_length: int = 128,
        preprocess_state: Optional[TextClassificationState] = None,
        label_to_class_mapping: Optional[dict] = None,
        batch_size: int = 16,
        num_workers: Optional[int] = None,
    ) -> 'TextClassificationData':
        """Creates a TextClassificationData object from files.

        Args:

            predict_file: Path to training data.
            input: The field storing the text to be classified.
            filetype: .csv or .json
            backbone: Tokenizer backbone to use, can use any HuggingFace tokenizer.
            batch_size: the batchsize to use for parallel loading. Defaults to 64.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to None which equals the number of available CPU threads,
                or 0 for Darwin platform.
        """
        return cls.from_files(
            None,
            input=input,
            target=None,
            filetype=filetype,
            backbone=backbone,
            val_file=None,
            test_file=None,
            predict_file=predict_file,
            max_length=max_length,
            label_to_class_mapping=label_to_class_mapping,
            batch_size=batch_size,
            num_workers=num_workers,
            preprocess_state=preprocess_state,
        )
