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
import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Generic, Iterable, List, Mapping, Optional, Sequence, Sized, Tuple, Type, TypeVar, Union

import numpy as np
from pytorch_lightning.trainer.states import RunningStage
from torch.nn import Module
from torchvision.datasets.folder import has_file_allowed_extension, make_dataset

from flash.data.auto_dataset import AutoDataset, BaseAutoDataset, IterableAutoDataset
from flash.data.data_pipeline import DataPipeline
from flash.data.process import ProcessState, Properties
from flash.data.utils import _STAGES_PREFIX, CurrentRunningStageFuncContext


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


class DataSource(Properties, Module, ABC):

    def __init__(
        self,
        train_data: Optional[Any] = None,
        val_data: Optional[Any] = None,
        test_data: Optional[Any] = None,
        predict_data: Optional[Any] = None,
    ):
        super().__init__()

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.predict_data = predict_data

    @abstractmethod
    def load_data(
        self,
        data: Any,
        dataset: Optional[Any] = None
    ) -> Iterable[Mapping[str, Any]]:  # TODO: decide what type this should be
        """Loads entire data from Dataset. The input ``data`` can be anything, but you need to return a Mapping.

        Example::

            # data: "."
            # output: [("./cat/1.png", 1), ..., ("./dog/10.png", 0)]

            output: Mapping = load_data(data)

        """

    @abstractmethod
    def load_sample(self, sample: Mapping[str, Any], dataset: Optional[Any] = None) -> Any:
        """Loads single sample from dataset"""

    def to_datasets(self, data_pipeline: DataPipeline) -> Tuple[Optional[BaseAutoDataset], ...]:
        train_dataset = self._generate_dataset_if_possible(RunningStage.TRAINING, data_pipeline)
        val_dataset = self._generate_dataset_if_possible(RunningStage.VALIDATING, data_pipeline)
        test_dataset = self._generate_dataset_if_possible(RunningStage.TESTING, data_pipeline)
        predict_dataset = self._generate_dataset_if_possible(RunningStage.PREDICTING, data_pipeline)
        return train_dataset, val_dataset, test_dataset, predict_dataset

    def _generate_dataset_if_possible(
        self,
        running_stage: RunningStage,
        data_pipeline: DataPipeline,
    ) -> Optional[Union[AutoDataset, IterableAutoDataset]]:
        data = getattr(self, f"{_STAGES_PREFIX[running_stage]}_data", None)
        if data is not None:
            return self.generate_dataset(data, running_stage, data_pipeline)

    def generate_dataset(
        self,
        data,
        running_stage: RunningStage,
        data_pipeline: DataPipeline,
    ) -> Optional[Union[AutoDataset, IterableAutoDataset]]:
        mock_dataset = MockDataset()
        with CurrentRunningStageFuncContext(running_stage, "load_data", self):
            data = self.load_data(data, mock_dataset)  # TODO: Should actually resolve this

        if has_len(data):
            dataset = AutoDataset(data, self, running_stage, data_pipeline)
        else:
            dataset = IterableAutoDataset(data, self, running_stage, data_pipeline)
        dataset.__dict__.update(mock_dataset.metadata)
        return dataset


T = TypeVar("T")


class DefaultDataSource(Enum):  # TODO: This could be replaced with a data source registry that the user can add to

    FOLDERS = "folders"
    FILES = "files"

    def as_type(self) -> Type[DataSource]:
        _data_source_types = {
            DefaultDataSource.FOLDERS: FoldersDataSource,
            DefaultDataSource.FILES: FilesDataSource,
        }
        return _data_source_types[self]


class SequenceDataSource(DataSource, ABC):

    def __init__(
        self,
        train_inputs: Optional[Sequence[Any]] = None,
        train_targets: Optional[Sequence[Any]] = None,
        val_inputs: Optional[Sequence[Any]] = None,
        val_targets: Optional[Sequence[Any]] = None,
        test_inputs: Optional[Sequence[Any]] = None,
        test_targets: Optional[Sequence[Any]] = None,
        predict_inputs: Optional[Sequence[Any]] = None,
        labels: Optional[Sequence[str]] = None
    ):
        super().__init__(
            train_data=(train_inputs, train_targets),
            val_data=(val_inputs, val_targets),
            test_data=(test_inputs, test_targets),
            predict_data=(predict_inputs, None),
        )

        self.labels = labels

        if self.labels is not None:
            self.set_state(LabelsState(self.labels))

    def load_data(self, data: Any, dataset: Optional[Any] = None) -> Iterable[Mapping[str, Any]]:
        inputs, targets = data
        if targets is None:
            return [{'input': input} for input in inputs]
        return [{'input': input, 'target': target} for input, target in zip(inputs, targets)]


class FoldersDataSource(DataSource, ABC):

    def __init__(
        self,
        train_folder: Optional[Union[str, pathlib.Path, list]] = None,
        val_folder: Optional[Union[str, pathlib.Path, list]] = None,
        test_folder: Optional[Union[str, pathlib.Path, list]] = None,
        predict_folder: Optional[Union[str, pathlib.Path, list]] = None,
        extensions: Optional[Tuple[str, ...]] = None,
    ):
        super().__init__(
            train_data=train_folder,
            val_data=val_folder,
            test_data=test_folder,
            predict_data=predict_folder,
        )

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

    def load_data(self, data: Any, dataset: Optional[Any] = None) -> Iterable[Mapping[str, Any]]:
        classes, class_to_idx = self.find_classes(data)
        if not classes:
            files = [os.path.join(data, file) for file in os.listdir(data)]
            return [{
                'input': file
            } for file in filter(
                lambda file: has_file_allowed_extension(file, self.extensions),
                files,
            )]
        self.set_state(LabelsState(classes))
        dataset.num_classes = len(classes)
        data = make_dataset(data, class_to_idx, extensions=self.extensions)
        return [{'input': input, 'target': target} for input, target in data]


class FilesDataSource(DataSource, ABC):

    def __init__(
        self,
        train_files: Optional[Union[Sequence[Union[str, pathlib.Path]], Iterable[Union[str, pathlib.Path]]]] = None,
        train_targets: Optional[Union[Sequence[Any], Iterable[Any]]] = None,
        val_files: Optional[Union[Sequence[Union[str, pathlib.Path]], Iterable[Union[str, pathlib.Path]]]] = None,
        val_targets: Optional[Union[Sequence[Any], Iterable[Any]]] = None,
        test_files: Optional[Union[Sequence[Union[str, pathlib.Path]], Iterable[Union[str, pathlib.Path]]]] = None,
        test_targets: Optional[Union[Sequence[Any], Iterable[Any]]] = None,
        predict_files: Optional[Union[Sequence[Union[str, pathlib.Path]], Iterable[Union[str, pathlib.Path]]]] = None,
        extensions: Optional[Tuple[str, ...]] = None,
    ):
        super().__init__(
            train_data=(train_files, train_targets),
            val_data=(val_files, val_targets),
            test_data=(test_files, test_targets),
            predict_data=(predict_files, None),
        )

        self.extensions = extensions

    def load_data(self, data: Any, dataset: Optional[Any] = None) -> Iterable[Mapping[str, Any]]:
        # TODO: Bring back the code to work out how many classes there are
        if isinstance(data, tuple):
            files, targets = data
        else:
            files, targets = data, None  # TODO: Sort this out
        if not targets:
            return [{'input': input} for input in files]
        return [{'input': file, 'target': target} for file, target in zip(files, targets)]
