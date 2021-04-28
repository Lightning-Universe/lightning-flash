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
from typing import Any, Dict, Generic, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from torch.nn import Module
from torchvision.datasets.folder import has_file_allowed_extension, make_dataset

from flash.data.process import ProcessState, Properties


@dataclass(unsafe_hash=True, frozen=True)
class LabelsState(ProcessState):

    labels: Optional[Sequence[str]]


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

    def train_load_data(self, dataset: Optional[Any] = None) -> Iterable[Mapping[str, Any]]:
        return self.load_data(self.train_data, dataset)

    def val_load_data(self, dataset: Optional[Any] = None) -> Iterable[Mapping[str, Any]]:
        return self.load_data(self.val_data, dataset)

    def test_load_data(self, dataset: Optional[Any] = None) -> Iterable[Mapping[str, Any]]:
        return self.load_data(self.test_data, dataset)

    def predict_load_data(self, dataset: Optional[Any] = None) -> Iterable[Mapping[str, Any]]:
        return self.load_data(self.predict_data, dataset)

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
        predict_targets: Optional[Sequence[Any]] = None,
        labels: Optional[Sequence[str]] = None
    ):
        super().__init__(
            train_data=(train_inputs, train_targets),
            val_data=(val_inputs, val_targets),
            test_data=(test_inputs, test_targets),
            predict_data=(predict_inputs, predict_targets),
        )

        self.labels = labels

        if self.labels is not None:
            self.set_state(LabelsState(self.labels))

    def load_data(self, data: Any, dataset: Optional[Any] = None) -> Iterable[Mapping[str, Any]]:
        inputs, targets = data
        if targets is None:
            return [{'input': input} for input in inputs]
        return [{'input': input, 'target': target} for input, target in zip(inputs, targets)]


class FolderDataSource(DataSource, ABC):

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
        self.set_state(LabelsState(classes))
        dataset.num_classes = len(classes)
        data = make_dataset(data, class_to_idx, extensions=self.extensions)
        return [{'input': input, 'target': target} for input, target in data]
