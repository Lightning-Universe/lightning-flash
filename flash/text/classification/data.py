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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from datasets import DatasetDict, load_dataset
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor
from transformers import AutoTokenizer, default_data_collator
from transformers.modeling_outputs import SequenceClassifierOutput

from flash.data.auto_dataset import AutoDataset
from flash.data.callback import BaseDataFetcher
from flash.data.data_module import DataModule
from flash.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources, LabelsState
from flash.data.process import Postprocess, Preprocess


class TextDataSource(DataSource):

    def __init__(self, backbone: str = "prajjwal1/bert-tiny", max_length: int = 128):
        super().__init__()

        self.backbone = backbone
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)

    def _tokenize_fn(
        self,
        ex: Union[Dict[str, str], str],
        input: str = None,
    ) -> Callable:
        """This function is used to tokenize sentences using the provided tokenizer."""
        if isinstance(ex, dict):
            ex = ex[input]
        return self.tokenizer(ex, max_length=self.max_length, truncation=True, padding="max_length")

    def _transform_label(self, label_to_class_mapping: Dict[str, int], target: str, ex: Dict[str, Union[int, str]]):
        ex[target] = label_to_class_mapping[ex[target]]
        return ex


class TextFileDataSource(TextDataSource):

    def __init__(self, filetype: str, backbone: str = "prajjwal1/bert-tiny", max_length: int = 128):
        super().__init__(backbone=backbone, max_length=max_length)

        self.filetype = filetype

    def load_data(
        self,
        data: Tuple[str, Union[str, List[str]], Union[str, List[str]]],
        dataset: Optional[Any] = None,
        columns: Union[List[str], Tuple[str]] = ("input_ids", "attention_mask", "labels"),
        use_full: bool = True,
    ) -> Union[Sequence[Mapping[str, Any]]]:
        csv_file, input, target = data

        data_files = {}

        stage = self.running_stage.value
        data_files[stage] = str(csv_file)

        # FLASH_TESTING is set in the CI to run faster.
        if use_full and os.getenv("FLASH_TESTING", "0") == "0":
            dataset_dict = load_dataset(self.filetype, data_files=data_files)
        else:
            # used for debugging. Avoid processing the entire dataset   # noqa E265
            dataset_dict = DatasetDict({
                stage: load_dataset(self.filetype, data_files=data_files, split=[f'{stage}[:20]'])[0]
            })

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

    def __init__(self, backbone: str = "prajjwal1/bert-tiny", max_length: int = 128):
        super().__init__("csv", backbone=backbone, max_length=max_length)


class TextJSONDataSource(TextFileDataSource):

    def __init__(self, backbone: str = "prajjwal1/bert-tiny", max_length: int = 128):
        super().__init__("json", backbone=backbone, max_length=max_length)


class TextSentencesDataSource(TextDataSource):

    def __init__(self, backbone: str = "prajjwal1/bert-tiny", max_length: int = 128):
        super().__init__(backbone=backbone, max_length=max_length)

    def load_data(
        self,
        data: Union[str, List[str]],
        dataset: Optional[Any] = None,
    ) -> Union[Sequence[Mapping[str, Any]]]:

        if isinstance(data, str):
            data = [data]
        return [self._tokenize_fn(s, ) for s in data]


class TextClassificationPreprocess(Preprocess):

    data_sources = {
        DefaultDataSources.CSV: TextCSVDataSource,
        "sentences": TextSentencesDataSource,
    }

    def get_state_dict(self) -> Dict[str, Any]:
        return {}

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


class TextClassificationPostProcess(Postprocess):

    def per_batch_transform(self, batch: Any) -> Any:
        if isinstance(batch, SequenceClassifierOutput):
            batch = batch.logits
        return super().per_batch_transform(batch)


class TextClassificationData(DataModule):
    """Data Module for text classification tasks"""

    preprocess_cls = TextClassificationPreprocess
    postprocess_cls = TextClassificationPostProcess

    @classmethod
    def from_csv(
        cls,
        input_fields: Union[str, List[str]],
        target_fields: Optional[Union[str, List[str]]] = None,
        train_file: Optional[str] = None,
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        predict_file: Optional[str] = None,
        backbone: str = "prajjwal1/bert-tiny",
        max_length: int = 128,
        data_fetcher: BaseDataFetcher = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
    ) -> 'DataModule':
        return super().from_csv(
            input_fields,
            target_fields,
            train_file=train_file,
            val_file=val_file,
            test_file=test_file,
            predict_file=predict_file,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            data_source_kwargs=dict(
                backbone=backbone,
                max_length=max_length,
            ),
        )
