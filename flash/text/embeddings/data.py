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
from pandas.core.frame import DataFrame

import flash
from flash.core.data.auto_dataset import AutoDataset
from flash.core.data.io.input import DataKeys, Input, LabelsState
from flash.core.utilities.imports import _TEXT_AVAILABLE, requires

if _TEXT_AVAILABLE:
    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer
    


class TextInput(Input):
    @requires("text")
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
        return self.tokenizer(ex[input], max_length=self.max_length, truncation=True, padding="max_length")

    @staticmethod
    def _transform_label(label_to_class_mapping: Dict[str, int], target: str, ex: Dict[str, Union[int, str]]):
        ex[target] = label_to_class_mapping[ex[target]]
        return ex

    @staticmethod
    def _multilabel_target(targets: List[str], element: Dict[str, Any]) -> Dict[str, Any]:
        targets = [element.pop(target) for target in targets]
        element[DataKeys.TARGET] = targets
        return element

    def _to_hf_dataset(self, data) -> Sequence[Mapping[str, Any]]:
        """account for flash CI testing context."""
        hf_dataset, *other = self.to_hf_dataset(data)

        if flash._IS_TESTING and not torch.cuda.is_available():
            # NOTE: must subset in this way to return a Dataset
            hf_dataset = hf_dataset.select(range(20))

        return (hf_dataset, *other)

    def load_data(
        self,
        data: Tuple[str, Union[str, List[str]], Union[str, List[str]]],
        dataset: Optional[Any] = None,
    ) -> Sequence[Mapping[str, Any]]:
        """Loads data into HuggingFace datasets.Dataset."""

        hf_dataset, input, *other = self._to_hf_dataset(data)

        if not self.predicting:
            target: Union[str, List[str]] = other.pop()
            if isinstance(target, List):
                # multi-target
                dataset.multi_label = True
                hf_dataset = hf_dataset.map(partial(self._multilabel_target, target))  # NOTE: renames target column
                dataset.num_classes = len(target)
                self.set_state(LabelsState(target))
            else:
                dataset.multi_label = False
                if self.training:
                    labels = list(sorted(list(set(hf_dataset[target]))))
                    dataset.num_classes = len(labels)
                    self.set_state(LabelsState(labels))

                labels = self.get_state(LabelsState)

                # convert labels to ids (note: the target column get overwritten)
                if labels is not None:
                    labels = labels.labels
                    label_to_class_mapping = {v: k for k, v in enumerate(labels)}
                    hf_dataset = hf_dataset.map(partial(self._transform_label, label_to_class_mapping, target))

                # rename label column
                hf_dataset = hf_dataset.rename_column(target, DataKeys.TARGET)

        # remove extra columns
        extra_columns = set(hf_dataset.column_names) - {input, DataKeys.TARGET}
        hf_dataset = hf_dataset.remove_columns(extra_columns)

        # tokenize
        hf_dataset = hf_dataset.map(partial(self._tokenize_fn, input=input), batched=True, remove_columns=[input])

        # set format
        hf_dataset.set_format("torch")

        return hf_dataset

    def predict_load_data(self, data: Any, dataset: AutoDataset):
        return self.load_data(data, dataset)

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("tokenizer")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True)


class TextCSVInput(TextInput):
    def to_hf_dataset(self, data: Tuple[str, str, str]) -> Tuple[Sequence[Mapping[str, Any]], str, str]:
        file, *other = data
        dataset_dict = load_dataset("csv", data_files={"train": str(file)})
        return (dataset_dict["train"], *other)


class TextJSONInput(TextInput):
    def to_hf_dataset(self, data: Tuple[str, str, str, str]) -> Tuple[Sequence[Mapping[str, Any]], str, str]:
        file, *other, field = data
        dataset_dict = load_dataset("json", data_files={"train": str(file)}, field=field)
        return (dataset_dict["train"], *other)


class TextDataFrameInput(TextInput):
    def to_hf_dataset(self, data: Tuple[DataFrame, str, str]) -> Tuple[Sequence[Mapping[str, Any]], str, str]:
        df, *other = data
        hf_dataset = Dataset.from_pandas(df)
        return (hf_dataset, *other)


class TextParquetInput(TextInput):
    def to_hf_dataset(self, data: Tuple[str, str, str]) -> Tuple[Sequence[Mapping[str, Any]], str, str]:
        file, *other = data
        hf_dataset = Dataset.from_parquet(str(file))
        return (hf_dataset, *other)


class TextHuggingFaceDatasetInput(TextInput):
    def to_hf_dataset(self, data: Tuple[str, str, str]) -> Tuple[Sequence[Mapping[str, Any]], str, str]:
        hf_dataset, *other = data
        return (hf_dataset, *other)


class TextListInput(TextInput):
    def to_hf_dataset(
        self, data: Union[Tuple[List[str], List[str]], List[str]]
    ) -> Tuple[Sequence[Mapping[str, Any]], Optional[List[str]]]:

        if isinstance(data, tuple):
            input_list, target_list = data
            # NOTE: here we already deal with multilabels
            # NOTE: here we already rename to correct column names
            hf_dataset = Dataset.from_dict({DataKeys.INPUT: input_list, DataKeys.TARGET: target_list})
            return hf_dataset, target_list

        # predicting
        hf_dataset = Dataset.from_dict({DataKeys.INPUT: data})

        return (hf_dataset,)

    def load_data(
        self,
        data: Tuple[List[str], Union[List[Any], List[List[Any]]]],
        dataset: Optional[Any] = None,
    ) -> Sequence[Mapping[str, Any]]:

        hf_dataset, *other = self._to_hf_dataset(data)

        if not self.predicting:
            target_list = other.pop()
            if isinstance(target_list[0], List):
                # multi-target_list
                dataset.multi_label = True
                dataset.num_classes = len(target_list[0])
                self.set_state(LabelsState(target_list))
            else:
                dataset.multi_label = False
                if self.training:
                    labels = list(sorted(list(set(hf_dataset[DataKeys.TARGET]))))
                    dataset.num_classes = len(labels)
                    self.set_state(LabelsState(labels))

                labels = self.get_state(LabelsState)

                # convert labels to ids
                if labels is not None:
                    labels = labels.labels
                    label_to_class_mapping = {v: k for k, v in enumerate(labels)}
                    # happens in-place and keeps the target column name
                    hf_dataset = hf_dataset.map(partial(self._transform_label, label_to_class_mapping, DataKeys.TARGET))

        # tokenize
        hf_dataset = hf_dataset.map(partial(self._tokenize_fn, input=DataKeys.INPUT), batched=True)

        # set format
        hf_dataset = hf_dataset.remove_columns([DataKeys.INPUT])  # just leave the numerical columns
        hf_dataset.set_format("torch")

        return hf_dataset


