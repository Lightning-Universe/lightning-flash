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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

import flash
from flash.core.data.auto_dataset import AutoDataset
from flash.core.data.data_module import DataModule
from flash.core.data.data_source import DataSource, DefaultDataSources, LabelsState
from flash.core.data.process import Postprocess, Preprocess
from flash.core.utilities.imports import _TEXT_AVAILABLE

if _TEXT_AVAILABLE:
    from datasets import DatasetDict, load_dataset
    from transformers import AutoTokenizer, default_data_collator
    from transformers.modeling_outputs import SequenceClassifierOutput


class TextDataSource(DataSource):

    def __init__(self, backbone: str, max_length: int = 128):
        super().__init__()

        if not _TEXT_AVAILABLE:
            raise ModuleNotFoundError("Please, pip install -e '.[text]'")

        self.tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
        self.max_length = max_length

    def _tokenize_fn(
        self,
        ex: Union[Dict[str, str], str],
        input: Optional[str] = None,
    ) -> Callable:
        """This function is used to tokenize sentences using the provided tokenizer."""
        if isinstance(ex, dict):
            ex = ex[input]
        return self.tokenizer(ex, max_length=self.max_length, truncation=True, padding="max_length")

    def _transform_label(self, label_to_class_mapping: Dict[str, int], target: str, ex: Dict[str, Union[int, str]]):
        ex[target] = label_to_class_mapping[ex[target]]
        return ex


class TextFileDataSource(TextDataSource):

    def __init__(self, filetype: str, backbone: str, max_length: int = 128):
        super().__init__(backbone, max_length=max_length)

        self.filetype = filetype

    def load_data(
        self,
        data: Tuple[str, Union[str, List[str]], Union[str, List[str]]],
        dataset: Optional[Any] = None,
        columns: Union[List[str], Tuple[str]] = ("input_ids", "attention_mask", "labels"),
    ) -> Union[Sequence[Mapping[str, Any]]]:
        csv_file, input, target = data

        data_files = {}

        stage = self.running_stage.value
        data_files[stage] = str(csv_file)

        # FLASH_TESTING is set in the CI to run faster.
        # FLASH_TESTING is set in the CI to run faster.
        if flash._IS_TESTING and not torch.cuda.is_available():
            try:
                dataset_dict = DatasetDict({
                    stage: load_dataset(self.filetype, data_files=data_files, split=[f'{stage}[:20]'])[0]
                })
            except Exception:
                dataset_dict = load_dataset(self.filetype, data_files=data_files)
        else:
            dataset_dict = load_dataset(self.filetype, data_files=data_files)

        if self.training:
            labels = list(sorted(list(set(dataset_dict[stage][target]))))
            dataset.num_classes = len(labels)
            self.set_state(LabelsState(labels))

        labels = self.get_state(LabelsState)

        # convert labels to ids
        # if not self.predicting:
        if labels is not None:
            labels = labels.labels
            label_to_class_mapping = {v: k for k, v in enumerate(labels)}
            dataset_dict = dataset_dict.map(partial(self._transform_label, label_to_class_mapping, target))

        dataset_dict = dataset_dict.map(partial(self._tokenize_fn, input=input), batched=True)

        # Hugging Face models expect target to be named ``labels``.
        if not self.predicting and target != "labels":
            dataset_dict.rename_column_(target, "labels")

        dataset_dict.set_format("torch", columns=columns)

        return dataset_dict[stage]

    def predict_load_data(self, data: Any, dataset: AutoDataset):
        return self.load_data(data, dataset, columns=["input_ids", "attention_mask"])


class TextCSVDataSource(TextFileDataSource):

    def __init__(self, backbone: str, max_length: int = 128):
        super().__init__("csv", backbone, max_length=max_length)


class TextJSONDataSource(TextFileDataSource):

    def __init__(self, backbone: str, max_length: int = 128):
        super().__init__("json", backbone, max_length=max_length)


class TextSentencesDataSource(TextDataSource):

    def __init__(self, backbone: str, max_length: int = 128):
        super().__init__(backbone, max_length=max_length)

    def load_data(
        self,
        data: Union[str, List[str]],
        dataset: Optional[Any] = None,
    ) -> Union[Sequence[Mapping[str, Any]]]:

        if isinstance(data, str):
            data = [data]
        return [self._tokenize_fn(s, ) for s in data]


class TextClassificationPreprocess(Preprocess):

    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        backbone: str = "prajjwal1/bert-tiny",
        max_length: int = 128,
    ):

        if not _TEXT_AVAILABLE:
            raise ModuleNotFoundError("Please, pip install -e '.[text]'")

        self.backbone = backbone
        self.max_length = max_length

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources={
                DefaultDataSources.CSV: TextCSVDataSource(self.backbone, max_length=max_length),
                DefaultDataSources.JSON: TextJSONDataSource(self.backbone, max_length=max_length),
                "sentences": TextSentencesDataSource(self.backbone, max_length=max_length),
            },
            default_data_source="sentences",
        )

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            **self.transforms,
            "backbone": self.backbone,
            "max_length": self.max_length,
        }

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool):
        return cls(**state_dict)

    def per_batch_transform(self, batch: Any) -> Any:
        if "labels" not in batch:
            # todo: understand why an extra dimension has been added.
            if batch["input_ids"].dim() == 3:
                batch["input_ids"] = batch["input_ids"].squeeze(0)
        return batch

    def collate(self, samples: Any) -> Tensor:
        """Override to convert a set of samples to a batch"""
        if isinstance(samples, dict):
            samples = [samples]
        return default_data_collator(samples)


class TextClassificationPostprocess(Postprocess):

    def per_batch_transform(self, batch: Any) -> Any:
        if isinstance(batch, SequenceClassifierOutput):
            batch = batch.logits
        return super().per_batch_transform(batch)


class TextClassificationData(DataModule):
    """Data Module for text classification tasks"""

    preprocess_cls = TextClassificationPreprocess
    postprocess_cls = TextClassificationPostprocess
