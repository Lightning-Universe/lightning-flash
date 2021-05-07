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
from inspect import signature
from typing import Any, Dict, Generic, Iterable, List, Mapping, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import torch
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.enums import LightningEnum
from torch.nn import Module
from torchvision.datasets.folder import has_file_allowed_extension, make_dataset

from flash.data.auto_dataset import AutoDataset, BaseAutoDataset, IterableAutoDataset
from flash.data.properties import ProcessState, Properties
from flash.data.utils import CurrentRunningStageFuncContext


def has_len(data: Union[Sequence[Any], Iterable[Any]]) -> bool:
    try:
        len(data)
        return True
    except (TypeError, NotImplementedError):
        return False


@dataclass(unsafe_hash=True, frozen=True)
class LabelsState(ProcessState):

    labels: Optional[Sequence[str]]


class MockDataset:

    def __init__(self):
        self.metadata = {}

    def __setattr__(self, key, value):
        if key != 'metadata':
            self.metadata[key] = value
        else:
            object.__setattr__(self, key, value)


DATA_TYPE = TypeVar("DATA_TYPE")


class DataSource(Generic[DATA_TYPE], Properties, Module):

    def load_data(self,
                  data: DATA_TYPE,
                  dataset: Optional[Any] = None) -> Union[Sequence[Mapping[str, Any]], Iterable[Mapping[str, Any]]]:
        """Loads entire data from Dataset. The input ``data`` can be anything, but you need to return a Mapping.

        Example::

            # data: "."
            # output: [("./cat/1.png", 1), ..., ("./dog/10.png", 0)]

            output: Mapping = load_data(data)

        """
        return data

    def load_sample(self, sample: Mapping[str, Any], dataset: Optional[Any] = None) -> Any:
        """Loads single sample from dataset"""
        return sample

    def to_datasets(
        self,
        train_data: Optional[DATA_TYPE] = None,
        val_data: Optional[DATA_TYPE] = None,
        test_data: Optional[DATA_TYPE] = None,
        predict_data: Optional[DATA_TYPE] = None,
    ) -> Tuple[Optional[BaseAutoDataset], ...]:
        train_dataset = self.generate_dataset(train_data, RunningStage.TRAINING)
        val_dataset = self.generate_dataset(val_data, RunningStage.VALIDATING)
        test_dataset = self.generate_dataset(test_data, RunningStage.TESTING)
        predict_dataset = self.generate_dataset(predict_data, RunningStage.PREDICTING)
        return train_dataset, val_dataset, test_dataset, predict_dataset

    def generate_dataset(
        self,
        data: Optional[DATA_TYPE],
        running_stage: RunningStage,
    ) -> Optional[Union[AutoDataset, IterableAutoDataset]]:
        is_none = data is None
        # TODO: we should parse better the possible data types here.
        # Are `pata_paths` considered as Sequence ? for now it pass
        # the statement found in below.
        if isinstance(data, Sequence):
            is_none = data[0] is None

        if not is_none:
            from flash.data.data_pipeline import DataPipeline

            mock_dataset = MockDataset()
            with CurrentRunningStageFuncContext(running_stage, "load_data", self):
                load_data = getattr(
                    self, DataPipeline._resolve_function_hierarchy(
                        "load_data",
                        self,
                        running_stage,
                        DataSource,
                    )
                )
                parameters = signature(load_data).parameters
                if len(parameters) > 1 and "dataset" in parameters:  # TODO: This was DATASET_KEY before
                    data = load_data(data, mock_dataset)
                else:
                    data = load_data(data)

            if has_len(data):
                dataset = AutoDataset(data, self, running_stage)
            else:
                dataset = IterableAutoDataset(data, self, running_stage)
            dataset.__dict__.update(mock_dataset.metadata)
            return dataset


class DefaultDataSources(LightningEnum):

    PATHS = "paths"
    NUMPY = "numpy"
    TENSOR = "tensor"
    CSV = "csv"
    JSON = "json"

    # TODO: Create a FlashEnum class???
    def __hash__(self) -> int:
        return hash(self.value)


class DefaultDataKeys(LightningEnum):

    INPUT = "input"
    TARGET = "target"

    # TODO: Create a FlashEnum class???
    def __hash__(self) -> int:
        return hash(self.value)


SEQUENCE_DATA_TYPE = TypeVar("SEQUENCE_DATA_TYPE")


class SequenceDataSource(
    Generic[SEQUENCE_DATA_TYPE],
    DataSource[Tuple[Sequence[SEQUENCE_DATA_TYPE], Optional[Sequence[Any]]]],
):

    def __init__(self, labels: Optional[Sequence[str]] = None):
        super().__init__()

        self.labels = labels

        if self.labels is not None:
            self.set_state(LabelsState(self.labels))

    def load_data(
        self,
        data: Tuple[Sequence[SEQUENCE_DATA_TYPE], Optional[Sequence[Any]]],
        dataset: Optional[Any] = None,
    ) -> Sequence[Mapping[str, Any]]:
        # TODO: Bring back the code to work out how many classes there are
        inputs, targets = data
        if targets is None:
            return self.predict_load_data(data)
        return [{
            DefaultDataKeys.INPUT: input,
            DefaultDataKeys.TARGET: target
        } for input, target in zip(inputs, targets)]

    def predict_load_data(self, data: Sequence[SEQUENCE_DATA_TYPE]) -> Sequence[Mapping[str, Any]]:
        return [{DefaultDataKeys.INPUT: input} for input in data]


class PathsDataSource(SequenceDataSource):  # TODO: Sort out the typing here

    def __init__(self, extensions: Optional[Tuple[str, ...]] = None):
        super().__init__()

        self.extensions = extensions

    @staticmethod
    def find_classes(dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset. Ensures that no class is a subdirectory of another.

        Args:
            dir: Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    @staticmethod
    def isdir(data: Union[str, Tuple[List[str], List[Any]]]) -> bool:
        try:
            return os.path.isdir(data)
        except TypeError:
            return False

    def load_data(self,
                  data: Union[str, Tuple[List[str], List[Any]]],
                  dataset: Optional[Any] = None) -> Iterable[Mapping[str, Any]]:
        if self.isdir(data):
            classes, class_to_idx = self.find_classes(data)
            if not classes:
                return self.predict_load_data(data)
            else:
                self.set_state(LabelsState(classes))

            if dataset is not None:
                dataset.num_classes = len(classes)

            data = make_dataset(data, class_to_idx, extensions=self.extensions)
            return [{DefaultDataKeys.INPUT: input, DefaultDataKeys.TARGET: target} for input, target in data]
        return list(
            filter(
                lambda sample: has_file_allowed_extension(sample[DefaultDataKeys.INPUT], self.extensions),
                super().load_data(data, dataset),
            )
        )

    def predict_load_data(self,
                          data: Union[str, List[str]],
                          dataset: Optional[Any] = None) -> Iterable[Mapping[str, Any]]:
        if self.isdir(data):
            data = [os.path.join(data, file) for file in os.listdir(data)]

        if not isinstance(data, list):
            data = [data]

        return list(
            filter(
                lambda sample: has_file_allowed_extension(sample[DefaultDataKeys.INPUT], self.extensions),
                super().predict_load_data(data),
            )
        )


class TensorDataSource(SequenceDataSource[torch.Tensor]):
    """"""  # TODO: Some docstring here


class NumpyDataSource(SequenceDataSource[np.ndarray]):
    """"""  # TODO: Some docstring here
