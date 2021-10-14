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
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Generic,
    Iterable,
    Iterator,
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
import pandas as pd
import torch
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.enums import LightningEnum
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from flash.core.data.properties import ProcessState
from flash.core.data_v2.base_dataset import BaseDataset, FlashDataset
from flash.core.data_v2.transforms.input_transform import INPUT_TRANSFORM_TYPE, InputTransform
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE, lazy_import, requires

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


class InputFormat(LightningEnum):
    """The ``InputFormat`` enum contains the data source names used by all of the default ``from_*`` methods in
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


class InputDataKeys(LightningEnum):
    """The ``InputDataKeys`` enum contains the keys that are used by built-in data sources to refer to inputs and
    targets."""

    INPUT = "input"
    PREDS = "preds"
    TARGET = "target"
    METADATA = "metadata"

    # TODO: Create a FlashEnum class???
    def __hash__(self) -> int:
        return hash(self.value)


@dataclass
class InputStateContainer:

    example_input_array: Any = None
    args: Optional[Any] = None
    kwargs: Optional[Dict[str, Any]] = None
    state: Optional[Dict] = None
    input_transform: Optional[InputTransform] = None

    @classmethod
    def from_dataset(cls, dataset: Optional[BaseDataset]) -> "InputStateContainer":
        if dataset:
            return cls(
                example_input_array=dataset.example_input_array,
                args=dataset.args,
                kwargs=dataset.kwargs,
                state=dataset._state,
                input_transform=dataset.transform,
            )
        return cls()


@dataclass
class InputsStateContainer:

    train_input_state: InputStateContainer
    val_input_state: InputStateContainer
    test_input_state: InputStateContainer
    predict_input_state: InputStateContainer

    @classmethod
    def from_datasets(
        cls,
        train_dataset: Optional[BaseDataset],
        val_dataset: Optional[BaseDataset],
        test_dataset: Optional[BaseDataset],
        predict_dataset: Optional[BaseDataset],
    ) -> "InputsStateContainer":
        return cls(
            train_input_state=InputStateContainer.from_dataset(train_dataset),
            val_input_state=InputStateContainer.from_dataset(val_dataset),
            test_input_state=InputStateContainer.from_dataset(test_dataset),
            predict_input_state=InputStateContainer.from_dataset(predict_dataset),
        )


class BaseDataFormat(LightningEnum):
    """The base class for creating ``data_format`` for :class:`~flash.core.data_v2.base_dataset.FlashDataset`."""

    def __hash__(self) -> int:
        return hash(self.value)


DATA_TYPE = TypeVar("DATA_TYPE")
SEQUENCE_DATA_TYPE = TypeVar("SEQUENCE_DATA_TYPE")


class DatasetInput(FlashDataset[Dataset]):
    """The ``DatasetInput`` implements default behaviours for data sources which expect the input to
    :meth:`~flash.core.data_v2.base_dataset.FlashDataset.load_data` to be a :class:`torch.utils.data.dataset.Dataset`

    Args:
        labels: Optionally pass the labels as a mapping from class index to label string. These will then be set as the
            :class:`~flash.core.data.data_source.LabelsState`.
    """

    def load_sample(self, sample: Any) -> Mapping[str, Any]:
        if isinstance(sample, tuple) and len(sample) == 2:
            return {InputDataKeys.INPUT: sample[0], InputDataKeys.TARGET: sample[1]}
        return {InputDataKeys.INPUT: sample}


class SequenceInput(
    Generic[SEQUENCE_DATA_TYPE],
    FlashDataset[Tuple[Sequence[SEQUENCE_DATA_TYPE], Optional[Sequence]]],
):
    """The ``SequenceInput`` implements default behaviours for data sources which expect the input to
    :meth:`~flash.core.data_v2.base_dataset.FlashDataset.load_data` to be a sequence of tuples (``(input, target)``
    where target can be ``None``).

    Args:
        labels: Optionally pass the labels as a mapping from class index to label string. These will then be set as the
            :class:`~flash.core.data.data_source.LabelsState`.
    """

    def __init__(
        self, running_stage: RunningStage, transform: INPUT_TRANSFORM_TYPE, labels: Optional[Sequence[str]] = None
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
        return [{InputDataKeys.INPUT: input, InputDataKeys.TARGET: target} for input, target in zip(inputs, targets)]

    @staticmethod
    def predict_load_data(data: Sequence[SEQUENCE_DATA_TYPE]) -> Sequence[Mapping[str, Any]]:
        return [{InputDataKeys.INPUT: input} for input in data]


class PathsInput(SequenceInput):
    """The ``PathsInput`` implements default behaviours for data sources which expect the input to
    :meth:`~flash.core.data_v2.base_dataset.FlashDataset.load_data` to be either a directory with a subdirectory for
    each class or a tuple containing list of files and corresponding list of targets.

    Args:
        extensions: The file extensions supported by this data source (e.g. ``(".jpg", ".png")``).
        labels: Optionally pass the labels as a mapping from class index to label string. These will then be set as the
            :class:`~flash.core.data.data_source.LabelsState`.
    """

    def __init__(
        self,
        running_stage: RunningStage,
        transform: INPUT_TRANSFORM_TYPE,
        extensions: Optional[Tuple[str, ...]] = None,
        loader: Optional[Callable[[str], Any]] = None,
        labels: Optional[Sequence[str]] = None,
    ):
        super().__init__(running_stage=running_stage, transform=transform, labels=labels)

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
            return [{InputDataKeys.INPUT: input, InputDataKeys.TARGET: target} for input, target in data]
        else:
            self.num_classes = len(np.unique(data[1]))

        return list(
            filter(
                lambda sample: has_file_allowed_extension(sample[InputDataKeys.INPUT], self.extensions),
                super().load_data(data),
            )
        )

    def predict_load_data(self, data: Union[str, List[str]]) -> Sequence[Mapping[str, Any]]:
        if self.isdir(data):
            data = [os.path.join(data, file) for file in os.listdir(data)]

        if not isinstance(data, list):
            data = [data]

        data = [{InputDataKeys.INPUT: input} for input in data]

        return list(
            filter(
                lambda sample: has_file_allowed_extension(sample[InputDataKeys.INPUT], self.extensions),
                data,
            )
        )

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        path = sample[InputDataKeys.INPUT]

        if self.loader is not None:
            sample[InputDataKeys.INPUT] = self.loader(path)

        sample[InputDataKeys.METADATA] = {
            "filepath": path,
        }
        return sample


class DataFrameInput(FlashDataset[Tuple[pd.DataFrame, str, Union[str, List[str]], Optional[str], Optional[str]]]):
    def __init__(self, running_stage: RunningStage, transform: INPUT_TRANSFORM_TYPE, loader: Callable[[str], Any]):
        super().__init__(running_stage=running_stage, transform=transform)

        self.loader = loader

    @staticmethod
    def _walk_files(root: str) -> Iterator[str]:
        for root, _, files in os.walk(root):
            for file in files:
                yield os.path.join(root, file)

    @staticmethod
    def _default_resolver(root: str, id: str):
        if os.path.isabs(id):
            return id

        pattern = f"*{id}*"

        try:
            return str(next(Path(root).rglob(pattern)))
        except StopIteration:
            raise ValueError(
                f"Found no matches for pattern: {pattern} in directory: {root}. File IDs should uniquely identify the "
                "file to load."
            )

    @staticmethod
    def _resolve_file(resolver: Callable[[str, str], str], root: str, input_key: str, row: pd.Series) -> pd.Series:
        row[input_key] = resolver(root, row[input_key])
        return row

    @staticmethod
    def _resolve_target(label_to_class: Dict[str, int], target_key: str, row: pd.Series) -> pd.Series:
        row[target_key] = label_to_class[row[target_key]]
        return row

    @staticmethod
    def _resolve_multi_target(target_keys: List[str], row: pd.Series) -> pd.Series:
        row[target_keys[0]] = [row[target_key] for target_key in target_keys]
        return row

    def load_data(
        self,
        data: Tuple[pd.DataFrame, str, Union[str, List[str]], Optional[str], Optional[str]],
    ) -> Sequence[Mapping[str, Any]]:
        data, input_key, target_keys, root, resolver = data

        if isinstance(data, (str, Path)):
            data = str(data)
            data_frame = pd.read_csv(data)
            if root is None:
                root = os.path.dirname(data)
        else:
            data_frame = data

        if root is None:
            root = ""

        if resolver is None:
            warnings.warn("Using default resolver, this may take a while.", UserWarning)
            resolver = self._default_resolver

        tqdm.pandas(desc="Resolving files")
        data_frame = data_frame.progress_apply(partial(self._resolve_file, resolver, root, input_key), axis=1)

        if not self.predicting:
            if isinstance(target_keys, List):
                self.multi_label = True
                self.num_classes = len(target_keys)
                self.set_state(LabelsState(target_keys))
                data_frame = data_frame.apply(partial(self._resolve_multi_target, target_keys), axis=1)
                target_keys = target_keys[0]
            else:
                self.multi_label = False
                if self.training:
                    labels = list(sorted(data_frame[target_keys].unique()))
                    self.num_classes = len(labels)
                    self.set_state(LabelsState(labels))

                labels = self.get_state(LabelsState)

                if labels is not None:
                    labels = labels.labels
                    label_to_class = {v: k for k, v in enumerate(labels)}
                    data_frame = data_frame.apply(partial(self._resolve_target, label_to_class, target_keys), axis=1)

            return [
                {
                    InputDataKeys.INPUT: row[input_key],
                    InputDataKeys.TARGET: row[target_keys],
                }
                for _, row in data_frame.iterrows()
            ]
        return [
            {
                InputDataKeys.INPUT: row[input_key],
            }
            for _, row in data_frame.iterrows()
        ]

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: simplify this duplicated code from PathsInput
        path = sample[InputDataKeys.INPUT]

        if self.loader is not None:
            sample[InputDataKeys.INPUT] = self.loader(path)

        sample[InputDataKeys.METADATA] = {
            "filepath": path,
        }
        return sample


class TensorInput(FlashDataset[torch.Tensor]):
    """The ``TensorInput`` is a ``SequenceInput`` which expects the input to
    :meth:`~flash.core.data_v2.base_dataset.FlashDataset.load_data` to be a sequence of ``torch.Tensor`` objects."""

    def load_data(
        self,
        data: Tuple[Sequence[SEQUENCE_DATA_TYPE], Optional[Sequence]],
    ) -> Sequence[Mapping[str, Any]]:
        # TODO: Bring back the code to work out how many classes there are
        if len(data) == 2:
            self.num_classes = len(torch.unique(torch.tensor(data[1])))
        return super().load_data(data)


class NumpyInput(SequenceInput[np.ndarray]):
    """The ``NumpyInput`` is a ``SequenceInput`` which expects the input to
    :meth:`~flash.core.data_v2.base_dataset.FlashDataset.load_data` to be a sequence of ``np.ndarray`` objects."""


class FiftyOneInput(FlashDataset[SampleCollection]):
    """The ``FiftyOneInput`` expects the input to
    :meth:`~flash.core.data_v2.base_dataset.FlashDataset.load_data`
    to be a ``fiftyone.core.collections.SampleCollection``."""

    def __init__(self, running_stage: RunningStage, transform: INPUT_TRANSFORM_TYPE, label_field: str = "ground_truth"):
        super().__init__(running_stage=running_stage, transform=transform)
        self.label_field = label_field

    @property
    @requires("fiftyone")
    def label_cls(self):
        return fol.Label

    @requires("fiftyone")
    def load_data(self, data: SampleCollection) -> Sequence[Mapping[str, Any]]:
        self._validate(data)

        label_path = data._get_label_field_path(self.label_field, "label")[1]

        filepaths = data.values("filepath")
        targets = data.values(label_path)

        classes = self._get_classes(data)

        self.num_classes = len(classes)

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        if targets and isinstance(targets[0], list):

            def to_idx(t):
                return [class_to_idx[x] for x in t]

        else:

            def to_idx(t):
                return class_to_idx[t]

        return [
            {
                InputDataKeys.INPUT: f,
                InputDataKeys.TARGET: to_idx(t),
            }
            for f, t in zip(filepaths, targets)
        ]

    @staticmethod
    @requires("fiftyone")
    def predict_load_data(data: SampleCollection) -> Sequence[Mapping[str, Any]]:
        return [{InputDataKeys.INPUT: f} for f in data.values("filepath")]

    def _validate(self, data):
        label_type = data._get_label_field_type(self.label_field)
        if not issubclass(label_type, self.label_cls):
            raise ValueError(f"Expected field '{self.label_field}' to have type {self.label_cls}; found {label_type}")

    def _get_classes(self, data):
        classes = data.classes.get(self.label_field, None)

        if not classes:
            classes = data.default_classes

        if not classes:
            label_path = data._get_label_field_path(self.label_field, "label")[1]
            classes = data.distinct(label_path)

        return classes
