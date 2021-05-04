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
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Generic, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union

from pytorch_lightning.trainer.states import RunningStage
from torch.nn import Module
from torchvision.datasets.folder import has_file_allowed_extension, make_dataset

from flash.data.auto_dataset import AutoDataset, BaseAutoDataset, IterableAutoDataset
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


DATA_TYPE = TypeVar('DATA_TYPE')


class DataSource(Generic[DATA_TYPE], Properties, Module, ABC):

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
        if data is not None:
            from flash.data.data_pipeline import DataPipeline

            mock_dataset = MockDataset()
            with CurrentRunningStageFuncContext(running_stage, "load_data", self):
                load_data = getattr(
                    self, DataPipeline._resolve_function_hierarchy(
                        'load_data',
                        self,
                        running_stage,
                        DataSource,
                    )
                )
                data = load_data(data, mock_dataset)

            if has_len(data):
                dataset = AutoDataset(data, self, running_stage)
            else:
                dataset = IterableAutoDataset(data, self, running_stage)
            dataset.__dict__.update(mock_dataset.metadata)
            return dataset


class DefaultDataSource(Enum):  # TODO: This could be replaced with a data source registry that the user can add to

    FOLDERS = "folders"
    FILES = "files"

    def as_type(self) -> Type[DataSource]:
        _data_source_types = {
            DefaultDataSource.FOLDERS: FoldersDataSource,
            DefaultDataSource.FILES: FilesDataSource,
        }
        return _data_source_types[self]


class FoldersDataSource(DataSource[str], ABC):

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

    def load_data(self, data: str, dataset: Optional[Any] = None) -> Iterable[Mapping[str, Any]]:
        classes, class_to_idx = self.find_classes(data)
        if not classes:
            files = [os.path.join(data, file) for file in os.listdir(data)]
            return [{
                'input': file
            } for file in filter(
                lambda file: has_file_allowed_extension(file, self.extensions),
                files,
            )]
        else:
            self.set_state(LabelsState(classes))
        dataset.num_classes = len(classes)
        data = make_dataset(data, class_to_idx, extensions=self.extensions)
        return [{'input': input, 'target': target} for input, target in data]


class FilesDataSource(DataSource[Tuple[Sequence[str], Optional[Sequence[Any]]]], ABC):

    def __init__(self, extensions: Optional[Tuple[str, ...]] = None):
        super().__init__()

        self.extensions = extensions

    def load_data(
        self,
        data: Tuple[Sequence[str], Optional[Sequence[Any]]],
        dataset: Optional[Any] = None,
    ) -> Iterable[Mapping[str, Any]]:
        # TODO: Bring back the code to work out how many classes there are
        if isinstance(data, tuple):
            files, targets = data
        else:
            files, targets = data, None  # TODO: Sort this out
        if not targets:
            return [{'input': input} for input in files]
        return [{'input': file, 'target': target} for file, target in zip(files, targets)]
