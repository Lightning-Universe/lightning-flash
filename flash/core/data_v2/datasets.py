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
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

import numpy as np
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.enums import LightningEnum
from torch.utils.data.dataset import Dataset

from flash.core.data.properties import ProcessState
from flash.core.data_v2.base_dataset import FlashDataset
from flash.core.data_v2.preprocess_transform import PreprocessTransform
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE, lazy_import

SampleCollection = None
if _FIFTYONE_AVAILABLE:
    fol = lazy_import("fiftyone.core.labels")
    if TYPE_CHECKING:
        from fiftyone.core.collections import SampleCollection
else:
    fol = None


# Credit to the PyTorchVision Team:
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py#L10
def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


# Credit to the PyTorchVision Team:
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py#L48
def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    Args:
        directory (str): root dataset directory
        class_to_idx (Dict[str, int]): dictionary mapping class name to class index
        extensions (optional): A list of allowed extensions.
            Either extensions or is_valid_file should be passed. Defaults to None.
        is_valid_file (optional): A function that takes path of a file
            and checks if the file is a valid file
            (used to check of corrupt files) both extensions and
            is_valid_file should not be passed. Defaults to None.

    Raises:
        ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.

    Returns:
        List[Tuple[str, int]]: samples of a form (path_to_sample, class)
    """
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances


def has_len(data: Union[Sequence[Any], Iterable[Any]]) -> bool:
    try:
        len(data)
        return True
    except (TypeError, NotImplementedError):
        return False


@dataclass(unsafe_hash=True, frozen=True)
class LabelsState(ProcessState):
    """A :class:`~flash.core.data.properties.ProcessState` containing ``labels``, a mapping from class index to
    label."""

    labels: Optional[Sequence[str]]


@dataclass(unsafe_hash=True, frozen=True)
class ImageLabelsMap(ProcessState):

    labels_map: Optional[Dict[int, Tuple[int, int, int]]]


class DefaultDataSources(LightningEnum):
    """The ``DefaultDataSources`` enum contains the data source names used by all of the default ``from_*`` methods in
    :class:`~flash.core.data.data_module.DataModule`."""

    FOLDERS = "folders"
    FILES = "files"
    NUMPY = "numpy"
    TENSORS = "tensors"
    CSV = "csv"
    JSON = "json"
    DATASETS = "datasets"
    FIFTYONE = "fiftyone"
    DATAFRAME = "data_frame"
    LISTS = "lists"
    SENTENCES = "sentences"
    LABELSTUDIO = "labelstudio"

    # TODO: Create a FlashEnum class???
    def __hash__(self) -> int:
        return hash(self.value)


class DefaultDataKeys(LightningEnum):
    """The ``DefaultDataKeys`` enum contains the keys that are used by built-in data sources to refer to inputs and
    targets."""

    INPUT = "input"
    PREDS = "preds"
    TARGET = "target"
    METADATA = "metadata"

    # TODO: Create a FlashEnum class???
    def __hash__(self) -> int:
        return hash(self.value)


class BaseDataFormat(LightningEnum):
    """The base class for creating ``data_format`` for :class:`~flash.core.data.data_source.DataSource`."""

    def __hash__(self) -> int:
        return hash(self.value)


class MockDataset:
    """The ``MockDataset`` catches any metadata that is attached through ``__setattr__``.

    This is passed to
    :meth:`~flash.core.data.data_source.DataSource.load_data` so that attributes can be set on the generated
    data set.
    """

    def __init__(self):
        self.metadata = {}

    def __setattr__(self, key, value):
        if key != "metadata":
            self.metadata[key] = value
        object.__setattr__(self, key, value)


SEQUENCE_DATA_TYPE = TypeVar("SEQUENCE_DATA_TYPE")


class DatasetDataSource(FlashDataset[Dataset]):
    """The ``DatasetDataSource`` implements default behaviours for data sources which expect the input to
    :meth:`~flash.core.data.data_source.DataSource.load_data` to be a :class:`torch.utils.data.dataset.Dataset`

    Args:
        labels: Optionally pass the labels as a mapping from class index to label string. These will then be set as the
            :class:`~flash.core.data.data_source.LabelsState`.
    """

    def load_sample(self, sample: Any, dataset: Optional[Any] = None) -> Mapping[str, Any]:
        if isinstance(sample, tuple) and len(sample) == 2:
            return {DefaultDataKeys.INPUT: sample[0], DefaultDataKeys.TARGET: sample[1]}
        return {DefaultDataKeys.INPUT: sample}


class SequenceDataset(
    Generic[SEQUENCE_DATA_TYPE],
    FlashDataset[Tuple[Sequence[SEQUENCE_DATA_TYPE], Optional[Sequence]]],
):
    """The ``SequenceDataset`` implements default behaviours for data sources which expect the input to
    :meth:`~flash.core.data.data_source.DataSource.load_data` to be a sequence of tuples (``(input, target)``
    where target can be ``None``).

    Args:
        labels: Optionally pass the labels as a mapping from class index to label string. These will then be set as the
            :class:`~flash.core.data.data_source.LabelsState`.
    """

    def __init__(
        self,
        running_stage: RunningStage,
        labels: Optional[Sequence[str]] = None,
        transform: Optional[PreprocessTransform] = None,
    ):
        super().__init__(running_stage=running_stage, transform=transform)

        self.labels = labels

        if self.labels is not None:
            self.set_state(LabelsState(self.labels))

    def load_data(
        self,
        data: Tuple[Sequence[SEQUENCE_DATA_TYPE], Optional[Sequence]],
    ) -> Sequence[Mapping[str, Any]]:
        # TODO: Bring back the code to work out how many classes there are
        inputs, targets = data
        if targets is None:
            return self.predict_load_data(data)
        return [
            {DefaultDataKeys.INPUT: input, DefaultDataKeys.TARGET: target} for input, target in zip(inputs, targets)
        ]

    @staticmethod
    def predict_load_data(data: Sequence[SEQUENCE_DATA_TYPE]) -> Sequence[Mapping[str, Any]]:
        return [{DefaultDataKeys.INPUT: input} for input in data]


class PathsDataset(SequenceDataset):
    """The ``PathsDataSource`` implements default behaviours for data sources which expect the input to
    :meth:`~flash.core.data.data_source.DataSource.load_data` to be either a directory with a subdirectory for
    each class or a tuple containing list of files and corresponding list of targets.

    Args:
        extensions: The file extensions supported by this data source (e.g. ``(".jpg", ".png")``).
        labels: Optionally pass the labels as a mapping from class index to label string. These will then be set as the
            :class:`~flash.core.data.data_source.LabelsState`.
    """

    def __init__(
        self,
        running_stage: RunningStage,
        extensions: Optional[Tuple[str, ...]] = None,
        loader: Optional[Callable[[str], Any]] = None,
        labels: Optional[Sequence[str]] = None,
        transform: Optional[PreprocessTransform] = None,
    ):
        super().__init__(labels=labels, running_stage=running_stage, transform=transform)

        self.extensions = extensions
        self.loader = loader

    @staticmethod
    def find_classes(dir: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset. Ensures that no class is a subdirectory of another.

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
            # data is not path-like (e.g. it may be a list of paths)
            return False

    def load_data(self, data: Union[str, Tuple[List[str], List[Any]]]) -> Sequence[Mapping[str, Any]]:
        if self.isdir(data):
            classes, class_to_idx = self.find_classes(data)
            if not classes:
                return self.predict_load_data(data)
            self.set_state(LabelsState(classes))

            self.num_classes = len(classes)

            data = make_dataset(data, class_to_idx, extensions=self.extensions)
            return [{DefaultDataKeys.INPUT: input, DefaultDataKeys.TARGET: target} for input, target in data]
        else:
            self.num_classes = len(np.unique(data[1]))

        return list(
            filter(
                lambda sample: has_file_allowed_extension(sample[DefaultDataKeys.INPUT], self.extensions),
                super().load_data(data),
            )
        )

    def predict_load_data(self, data: Union[str, List[str]]) -> Sequence[Mapping[str, Any]]:
        if self.isdir(data):
            data = [os.path.join(data, file) for file in os.listdir(data)]

        if not isinstance(data, list):
            data = [data]

        data = [{DefaultDataKeys.INPUT: input} for input in data]

        return list(
            filter(
                lambda sample: has_file_allowed_extension(sample[DefaultDataKeys.INPUT], self.extensions),
                data,
            )
        )

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        path = sample[DefaultDataKeys.INPUT]

        if self.loader is not None:
            sample[DefaultDataKeys.INPUT] = self.loader(path)

        sample[DefaultDataKeys.METADATA] = {
            "filepath": path,
        }
        return sample
