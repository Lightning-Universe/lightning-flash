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
from flash.core.data.process import Deserializer, Postprocess, Preprocess
from flash.core.utilities.imports import _TEXT_AVAILABLE, requires_extras

if _TEXT_AVAILABLE:
    from datasets import DatasetDict, load_dataset
    from transformers import AutoTokenizer, default_data_collator
    from transformers.modeling_outputs import SequenceClassifierOutput


class TextDeserializer(Deserializer):
    @requires_extras("text")
    def __init__(self, backbone: str, max_length: int, use_fast: bool = True):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=use_fast)
        self.max_length = max_length

    def deserialize(self, text: str) -> Tensor:
        return self.tokenizer(text, max_length=self.max_length, truncation=True, padding="max_length")

    @property
    def example_input(self) -> str:
        return "An example input"

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("tokenizer")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True)


class TextDataSource(DataSource):
    @requires_extras("text")
    def __init__(self, backbone: str, max_length: int = 128):
        super().__init__()

        self.backbone = backbone
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

    @staticmethod
    def _transform_label(label_to_class_mapping: Dict[str, int], target: str, ex: Dict[str, Union[int, str]]):
        ex[target] = label_to_class_mapping[ex[target]]
        return ex

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("tokenizer")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True)


class TextFileDataSource(TextDataSource):
    def __init__(self, filetype: str, backbone: str, max_length: int = 128):
        super().__init__(backbone, max_length=max_length)

        self.filetype = filetype

    @staticmethod
    def _multilabel_target(targets, element):
        targets = [element.pop(target) for target in targets]
        element["labels"] = targets
        return element

    def load_data(
        self,
        data: Tuple[str, Union[str, List[str]], Union[str, List[str]]],
        dataset: Optional[Any] = None,
        columns: Union[List[str], Tuple[str]] = ("input_ids", "attention_mask", "labels"),
    ) -> Union[Sequence[Mapping[str, Any]]]:
        if self.filetype == "json":
            file, input, target, field = data
        else:
            file, input, target = data

        data_files = {}

        stage = self.running_stage.value
        data_files[stage] = str(file)

        # FLASH_TESTING is set in the CI to run faster.
        if flash._IS_TESTING and not torch.cuda.is_available():
            try:
                if self.filetype == "json" and field is not None:
                    dataset_dict = DatasetDict(
                        {
                            stage: load_dataset(
                                self.filetype, data_files=data_files, split=[f"{stage}[:20]"], field=field
                            )[0]
                        }
                    )
                else:
                    dataset_dict = DatasetDict(
                        {stage: load_dataset(self.filetype, data_files=data_files, split=[f"{stage}[:20]"])[0]}
                    )
            except Exception:
                if self.filetype == "json" and field is not None:
                    dataset_dict = load_dataset(self.filetype, data_files=data_files, field=field)
                else:
                    dataset_dict = load_dataset(self.filetype, data_files=data_files)
        else:
            if self.filetype == "json" and field is not None:
                dataset_dict = load_dataset(self.filetype, data_files=data_files, field=field)
            else:
                dataset_dict = load_dataset(self.filetype, data_files=data_files)

        if not self.predicting:
            if isinstance(target, List):
                # multi-target
                dataset.multi_label = True
                dataset_dict = dataset_dict.map(partial(self._multilabel_target, target))
                dataset.num_classes = len(target)
                self.set_state(LabelsState(target))
            else:
                dataset.multi_label = False
                if self.training:
                    labels = list(sorted(list(set(dataset_dict[stage][target]))))
                    dataset.num_classes = len(labels)
                    self.set_state(LabelsState(labels))

                labels = self.get_state(LabelsState)

                # convert labels to ids
                if labels is not None:
                    labels = labels.labels
                    label_to_class_mapping = {v: k for k, v in enumerate(labels)}
                    dataset_dict = dataset_dict.map(partial(self._transform_label, label_to_class_mapping, target))

                # Hugging Face models expect target to be named ``labels``.
                if target != "labels":
                    dataset_dict.rename_column_(target, "labels")

        dataset_dict = dataset_dict.map(partial(self._tokenize_fn, input=input), batched=True)
        dataset_dict.set_format("torch", columns=columns)

        return dataset_dict[stage]

    def predict_load_data(self, data: Any, dataset: AutoDataset):
        return self.load_data(data, dataset, columns=["input_ids", "attention_mask"])

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("tokenizer")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True)


class TextCSVDataSource(TextFileDataSource):
    def __init__(self, backbone: str, max_length: int = 128):
        super().__init__("csv", backbone, max_length=max_length)

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("tokenizer")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True)


class TextJSONDataSource(TextFileDataSource):
    def __init__(self, backbone: str, max_length: int = 128):
        super().__init__("json", backbone, max_length=max_length)

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("tokenizer")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True)


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
        return [
            self._tokenize_fn(
                s,
            )
            for s in data
        ]

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("tokenizer")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True)


class TextClassificationPreprocess(Preprocess):
    @requires_extras("text")
    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        backbone: str = "prajjwal1/bert-tiny",
        max_length: int = 128,
    ):
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
            deserializer=TextDeserializer(backbone, max_length),
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
        """Override to convert a set of samples to a batch."""
        if isinstance(samples, dict):
            samples = [samples]
        return default_data_collator(samples)


class TextClassificationPostprocess(Postprocess):
    def per_batch_transform(self, batch: Any) -> Any:
        if isinstance(batch, SequenceClassifierOutput):
            batch = batch.logits
        return super().per_batch_transform(batch)


class TextClassificationData(DataModule):
    """Data Module for text classification tasks."""

    preprocess_cls = TextClassificationPreprocess
    postprocess_cls = TextClassificationPostprocess

    @property
    def backbone(self) -> Optional[str]:
        return getattr(self.preprocess, "backbone", None)
