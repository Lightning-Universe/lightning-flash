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
import typing
import warnings
from dataclasses import dataclass
from functools import partial
from inspect import signature
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
from torch.nn import Module
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from flash.core.data.auto_dataset import AutoDataset, BaseAutoDataset, IterableAutoDataset
from flash.core.data.properties import ProcessState, Properties
from flash.core.data.utils import CurrentRunningStageFuncContext
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


DATA_TYPE = TypeVar("DATA_TYPE")


class DataSource(Generic[DATA_TYPE], Properties, Module):
    """The ``DataSource`` class encapsulates two hooks: ``load_data`` and ``load_sample``.

    The
    :meth:`~flash.core.data.data_source.DataSource.to_datasets` method can then be used to automatically construct data
    sets from the hooks.
    """

    @staticmethod
    def load_data(
        data: DATA_TYPE,
        dataset: Optional[Any] = None,
    ) -> Union[Sequence[Mapping[str, Any]], Iterable[Mapping[str, Any]]]:
        """Given the ``data`` argument, the ``load_data`` hook produces a sequence or iterable of samples or
        sample metadata. The ``data`` argument can be anything, but this method should return a sequence or iterable of
        mappings from string (e.g. "input", "target", "bbox", etc.) to data (e.g. a target value) or metadata (e.g. a
        filename). Where possible, any heavy data loading should be performed in
        :meth:`~flash.core.data.data_source.DataSource.load_sample`. If the output is an iterable rather than a sequence
        (that is, it doesn't have length) then the generated dataset will be an ``IterableDataset``.

        Args:
            data: The data required to load the sequence or iterable of samples or sample metadata.
            dataset: Overriding methods can optionally include the dataset argument. Any attributes set on the dataset
                (e.g. ``num_classes``) will also be set on the generated dataset.

        Returns:
            A sequence or iterable of samples or sample metadata to be used as inputs to
            :meth:`~flash.core.data.data_source.DataSource.load_sample`.

        Example::

            # data: "."
            # output: [{"input": "./cat/1.png", "target": 1}, ..., {"input": "./dog/10.png", "target": 0}]

            output: Sequence[Mapping[str, Any]] = load_data(data)

        """
        return data

    @staticmethod
    def load_sample(sample: Mapping[str, Any], dataset: Optional[Any] = None) -> Any:
        """Given an element from the output of a call to
        :meth:`~flash.core.data.data_source.DataSource.load_data`, this hook
        should load a single data sample. The keys and values in the ``sample`` argument will be same as the keys and
        values in the outputs of :meth:`~flash.core.data.data_source.DataSource.load_data`.

        Args:
            sample: An element (sample or sample metadata) from the output of a call to
                :meth:`~flash.core.data.data_source.DataSource.load_data`.
            dataset: Overriding methods can optionally include the dataset argument. Any attributes set on the dataset
                (e.g. ``num_classes``) will also be set on the generated dataset.

        Returns:
            The loaded sample as a mapping with string keys (e.g. "input", "target") that can be processed by the
            :meth:`~flash.core.data.process.Preprocess.pre_tensor_transform`.

        Example::

            # sample: {"input": "./cat/1.png", "target": 1}
            # output: {"input": PIL.Image, "target": 1}

            output: Mapping[str, Any] = load_sample(sample)

        """
        return sample

    def to_datasets(
        self,
        train_data: Optional[DATA_TYPE] = None,
        val_data: Optional[DATA_TYPE] = None,
        test_data: Optional[DATA_TYPE] = None,
        predict_data: Optional[DATA_TYPE] = None,
    ) -> Tuple[Optional[BaseAutoDataset], ...]:
        """Construct data sets (of type :class:`~flash.core.data.auto_dataset.BaseAutoDataset`) from this data
        source by calling :meth:`~flash.core.data.data_source.DataSource.load_data` with each of the ``*_data``
        arguments. If an argument is given as ``None`` then no dataset will be created for that stage (``train``,
        ``val``, ``test``, ``predict``).

        Args:
            train_data: The input to :meth:`~flash.core.data.data_source.DataSource.load_data` to use to create the
                train dataset.
            val_data: The input to :meth:`~flash.core.data.data_source.DataSource.load_data` to use to create the
                validation dataset.
            test_data: The input to :meth:`~flash.core.data.data_source.DataSource.load_data` to use to create the
                test dataset.
            predict_data: The input to :meth:`~flash.core.data.data_source.DataSource.load_data` to use to create
                the predict dataset.

        Returns:
            A tuple of ``train_dataset``, ``val_dataset``, ``test_dataset``, ``predict_dataset``. If any ``*_data``
            argument is not passed to this method then the corresponding ``*_dataset`` will be ``None``.
        """
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
        """Generate a single dataset with the given input to
        :meth:`~flash.core.data.data_source.DataSource.load_data` for the given ``running_stage``.

        Args:
            data: The input to :meth:`~flash.core.data.data_source.DataSource.load_data` to use to create the dataset.
            running_stage: The running_stage for this dataset.

        Returns:
            The constructed :class:`~flash.core.data.auto_dataset.BaseAutoDataset`.
        """
        is_none = data is None

        if isinstance(data, Sequence):
            is_none = data[0] is None

        if not is_none:
            from flash.core.data.data_pipeline import DataPipeline

            mock_dataset = typing.cast(AutoDataset, MockDataset())
            with CurrentRunningStageFuncContext(running_stage, "load_data", self):
                resolved_func_name = DataPipeline._resolve_function_hierarchy(
                    "load_data", self, running_stage, DataSource
                )
                load_data: Callable[[DATA_TYPE, Optional[Any]], Any] = getattr(self, resolved_func_name)
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


SEQUENCE_DATA_TYPE = TypeVar("SEQUENCE_DATA_TYPE")


class DatasetDataSource(DataSource[Dataset]):
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


class SequenceDataSource(
    Generic[SEQUENCE_DATA_TYPE],
    DataSource[Tuple[Sequence[SEQUENCE_DATA_TYPE], Optional[Sequence]]],
):
    """The ``SequenceDataSource`` implements default behaviours for data sources which expect the input to
    :meth:`~flash.core.data.data_source.DataSource.load_data` to be a sequence of tuples (``(input, target)``
    where target can be ``None``).

    Args:
        labels: Optionally pass the labels as a mapping from class index to label string. These will then be set as the
            :class:`~flash.core.data.data_source.LabelsState`.
    """

    def __init__(self, labels: Optional[Sequence[str]] = None):
        super().__init__()

        self.labels = labels

        if self.labels is not None:
            self.set_state(LabelsState(self.labels))

    def load_data(
        self,
        data: Tuple[Sequence[SEQUENCE_DATA_TYPE], Optional[Sequence]],
        dataset: Optional[Any] = None,
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


class PathsDataSource(SequenceDataSource):
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
        extensions: Optional[Tuple[str, ...]] = None,
        loader: Optional[Callable[[str], Any]] = None,
        labels: Optional[Sequence[str]] = None,
    ):
        super().__init__(labels=labels)

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

    def load_data(
        self, data: Union[str, Tuple[List[str], List[Any]]], dataset: Optional[Any] = None
    ) -> Sequence[Mapping[str, Any]]:
        if self.isdir(data):
            classes, class_to_idx = self.find_classes(data)
            if not classes:
                return self.predict_load_data(data)
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

    def predict_load_data(
        self, data: Union[str, List[str]], dataset: Optional[Any] = None
    ) -> Sequence[Mapping[str, Any]]:
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

    def load_sample(self, sample: Dict[str, Any], dataset: Optional[Any] = None) -> Dict[str, Any]:
        path = sample[DefaultDataKeys.INPUT]

        if self.loader is not None:
            sample[DefaultDataKeys.INPUT] = self.loader(path)

        sample[DefaultDataKeys.METADATA] = {
            "filepath": path,
        }
        return sample


class LoaderDataFrameDataSource(
    DataSource[Tuple[pd.DataFrame, str, Union[str, List[str]], Optional[str], Optional[str]]]
):
    def __init__(self, loader: Callable[[str], Any]):
        super().__init__()

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
        dataset: Optional[Any] = None,
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
                dataset.multi_label = True
                dataset.num_classes = len(target_keys)
                self.set_state(LabelsState(target_keys))
                data_frame = data_frame.apply(partial(self._resolve_multi_target, target_keys), axis=1)
                target_keys = target_keys[0]
            else:
                dataset.multi_label = False
                if self.training:
                    labels = list(sorted(data_frame[target_keys].unique()))
                    dataset.num_classes = len(labels)
                    self.set_state(LabelsState(labels))

                labels = self.get_state(LabelsState)

                if labels is not None:
                    labels = labels.labels
                    label_to_class = {v: k for k, v in enumerate(labels)}
                    data_frame = data_frame.apply(partial(self._resolve_target, label_to_class, target_keys), axis=1)

            return [
                {
                    DefaultDataKeys.INPUT: row[input_key],
                    DefaultDataKeys.TARGET: row[target_keys],
                }
                for _, row in data_frame.iterrows()
            ]
        else:
            return [
                {
                    DefaultDataKeys.INPUT: row[input_key],
                }
                for _, row in data_frame.iterrows()
            ]

    def load_sample(self, sample: Dict[str, Any], dataset: Optional[Any] = None) -> Dict[str, Any]:
        # TODO: simplify this duplicated code from PathsDataSource
        path = sample[DefaultDataKeys.INPUT]

        if self.loader is not None:
            sample[DefaultDataKeys.INPUT] = self.loader(path)

        sample[DefaultDataKeys.METADATA] = {
            "filepath": path,
        }
        return sample


class TensorDataSource(SequenceDataSource[torch.Tensor]):
    """The ``TensorDataSource`` is a ``SequenceDataSource`` which expects the input to
    :meth:`~flash.core.data.data_source.DataSource.load_data` to be a sequence of ``torch.Tensor`` objects."""


class NumpyDataSource(SequenceDataSource[np.ndarray]):
    """The ``NumpyDataSource`` is a ``SequenceDataSource`` which expects the input to
    :meth:`~flash.core.data.data_source.DataSource.load_data` to be a sequence of ``np.ndarray`` objects."""


class FiftyOneDataSource(DataSource[SampleCollection]):
    """The ``FiftyOneDataSource`` expects the input to
    :meth:`~flash.core.data.data_source.DataSource.load_data` to be a ``fiftyone.core.collections.SampleCollection``."""

    def __init__(self, label_field: str = "ground_truth"):
        super().__init__()
        self.label_field = label_field

    @property
    @requires("fiftyone")
    def label_cls(self):
        return fol.Label

    @requires("fiftyone")
    def load_data(self, data: SampleCollection, dataset: Optional[Any] = None) -> Sequence[Mapping[str, Any]]:
        self._validate(data)

        label_path = data._get_label_field_path(self.label_field, "label")[1]

        filepaths = data.values("filepath")
        targets = data.values(label_path)

        classes = self._get_classes(data)

        if dataset is not None:
            dataset.num_classes = len(classes)

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        if targets and isinstance(targets[0], list):

            def to_idx(t):
                return [class_to_idx[x] for x in t]

        else:

            def to_idx(t):
                return class_to_idx[t]

        return [
            {
                DefaultDataKeys.INPUT: f,
                DefaultDataKeys.TARGET: to_idx(t),
            }
            for f, t in zip(filepaths, targets)
        ]

    @staticmethod
    @requires("fiftyone")
    def predict_load_data(data: SampleCollection, dataset: Optional[Any] = None) -> Sequence[Mapping[str, Any]]:
        return [{DefaultDataKeys.INPUT: f} for f in data.values("filepath")]

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
