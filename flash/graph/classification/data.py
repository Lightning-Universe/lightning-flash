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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import networkx as nx
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch_geometric.data import DataLoader, Dataset

from flash.data.base_viz import BaseVisualization  # for viz
from flash.data.callback import BaseDataFetcher
from flash.data.data_module import DataModule
from flash.data.data_source import DefaultDataKeys, DefaultDataSources
from flash.data.process import Preprocess
from flash.graph.classification.transforms import default_transforms, train_default_transforms
from flash.graph.data import ImageNumpyDataSource, ImagePathsDataSource, ImageTensorDataSource
from flash.utils.imports import _MATPLOTLIB_AVAILABLE
'''
[Deprecated]: The structure we follow is DataSet -> DataLoader -> DataModule -> DataPipeline
'''


class GraphClassificationPreprocess(Preprocess):

    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        num_features: int = 128
    ):
        self.num_features = num_features

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources={
                DefaultDataSources.FILES: GraphPathsDataSource(),
                DefaultDataSources.FOLDERS: GraphPathsDataSource(),
                DefaultDataSources.NUMPY: GraphNumpyDataSource(),
                DefaultDataSources.TENSORS: GraphTensorDataSource(),
            },
            default_data_source=DefaultDataSources.FILES,
        )

    def get_state_dict(self) -> Dict[str, Any]:
        return {**self.transforms, "num_features": self.num_features}

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = False):
        return cls(**state_dict)

    def default_transforms(self) -> Optional[Dict[str, Callable]]:
        return default_transforms(self.num_features)

    def train_default_transforms(self) -> Optional[Dict[str, Callable]]:
        return train_default_transforms(self.num_features)


class GraphClassificationData(DataModule):
    """Data module for graph classification tasks."""

    preprocess_cls = GraphClassificationPreprocess


# [DEPRECATED FROM HERE]


class BasicGraphDataset(Dataset):
    '''
    #todo: Probably unnecessary having the following class.
    '''

    def __init__(
        self, root=None, processed_dir='processed', raw_dir='raw', transform=None, pre_transform=None, pre_filter=None
    ):

        super(BasicGraphDataset, self).__init__(root, transform, pre_transform, pre_filter)

        self._processed_dir = processed_dir
        self._raw_dir = raw_dir

    @property
    def raw_dir(self):
        return os.path.join(self.root, self._raw_dir)

    @property
    def processed_dir(self):
        '''self.processed_dir already has root on it'''
        return os.path.join(self.root, self._processed_dir)

    @property
    def raw_file_names(self):
        '''self.raw_dir already has root on it'''
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return os.listdir(self.processed_dir)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        #TODO: Is data.pt the best way/file type to load the data?
        #TODO: Interface with networkx would probably go here with some option to say how to load it
        return data


class FilepathDataset(torch.utils.data.Dataset):
    """Dataset that takes in filepaths and labels. Taken from image"""

    def __init__(
        self,
        filepaths: Optional[Sequence[Union[str, pathlib.Path]]],
        labels: Optional[Sequence],
        loader: Callable = torch.load,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            filepaths: file paths to load with :attr:`loader`
            labels: the labels corresponding to the :attr:`filepaths`.
                Each unique value will get a class index by sorting them.
            loader: the function to load an graph from a given file path
            transform: the transforms to apply to the loaded graphs
        """
        self.fnames = filepaths or []
        self.labels = labels or []
        self.transform = transform
        self.loader = loader
        if self.has_labels:
            self.label_to_class_mapping = {v: k for k, v in enumerate(list(sorted(list(set(self.fnames)))))}

    @property
    def has_dict_labels(self) -> bool:
        return isinstance(self.labels, dict)

    @property
    def has_labels(self) -> bool:
        return self.labels is not None

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, index: int) -> Tuple[Any, Optional[int]]:
        filename = self.fnames[index]
        graph = self.loader(filename)
        label = None
        if self.has_dict_labels:
            name = os.path.splitext(filename)[0]
            name = os.path.basename(name)
            label = self.labels[name]
        if self.has_labels:
            label = self.label_to_class_mapping[filename]
        return graph, label


class FlashDatasetFolder(torch.utils.data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root: Root directory path.
        loader: A function to load a sample given its path.
        extensions: A list of allowed extensions. both extensions
            and is_valid_file should not be passed.
        transform: A function/transform that takes in
            a sample and returns a transformed version.
            E.g, https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html
        target_transform: A function/transform that takes
            in the target and transforms it.
        is_valid_file: A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
        with_targets: Whether to include targets
        graph_paths: List of graph paths to load. Only used when ``with_targets=False``

    Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each graph in the dataset
    """

    def __init__(
        self,
        root: str,
        loader: Callable,
        extensions: Tuple[
            str] = Graph_EXTENSIONS,  #todo: Graph_EXTENSIONS is not defined. In PyG the extension .pt is used
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable] = None,
        with_targets: bool = True,
        graph_paths: Optional[List[str]] = None,
    ):
        super(FlashDatasetFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        self.loader = loader
        self.extensions = extensions
        self.with_targets = with_targets

        if with_targets:
            classes, class_to_idx = self._find_classes(self.root)
            samples = self._make_dataset(self.root, class_to_idx, extensions, is_valid_file)
            if len(samples) == 0:
                msg = "Found 0 files in subfolders of: {}\n".format(self.root)
                if extensions is not None:
                    msg += "Supported extensions are: {}".format(",".join(extensions))
                raise RuntimeError(msg)

            self.classes = classes
            self.class_to_idx = class_to_idx
            self.samples = samples
            self.targets = [s[1] for s in samples]
        else:
            if not graph_paths:
                raise MisconfigurationException(
                    "`FlashDatasetFolder(with_target=False)` but no `graph_paths` were provided"
                )
            self.samples = graph_paths

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.with_targets:
            path, target = self.samples[index]
            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return (sample, target) if self.with_targets else sample

    def __len__(self) -> int:
        return len(self.samples)

    def _make_dataset(self, dir, class_to_idx):
        files = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if self._is_graph_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        files.append(item)

        return files

    def _is_graph_file(self, filename):
        return any(filename.endswith(extension) for extension in self.extensions)


class GraphClassificationData(DataModule):
    """Data module for graph classification tasks."""

    @classmethod
    def from_filepaths(
        cls,
        train_filepaths: Optional[Sequence[Union[str, pathlib.Path]]] = None,
        train_labels: Optional[Sequence] = None,
        train_transform: Optional[Callable] = None,
        valid_filepaths: Optional[Sequence[Union[str, pathlib.Path]]] = None,
        valid_labels: Optional[Sequence] = None,
        valid_transform: Optional[Callable] = None,
        test_filepaths: Optional[Sequence[Union[str, pathlib.Path]]] = None,
        test_labels: Optional[Sequence] = None,
        loader: Callable = torch.dataloader,
        batch_size: int = 64,
        num_workers: Optional[int] = None,
        **kwargs
    ):
        """Creates a GraphClassificationData object from lists of Graph filepaths and labels

        Args:
            train_filepaths: sequence of file paths for training dataset. Defaults to None.
            train_labels: sequence of labels for training dataset. Defaults to None.
            train_transform: transforms for training dataset. Defaults to None.
            valid_filepaths: sequence of file paths for validation dataset. Defaults to None.
            valid_labels: sequence of labels for validation dataset. Defaults to None.
            valid_transform: transforms for validation and testing dataset. Defaults to None.
            test_filepaths: sequence of file paths for test dataset. Defaults to None.
            test_labels: sequence of labels for test dataset. Defaults to None.
            loader: function to load an Graph file. Defaults to None.
            batch_size: the batchsize to use for parallel loading. Defaults to 64.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to None which equals the number of available CPU threads.

        Returns:
            GraphClassificationData: The constructed data module.

        Examples:
            >>> _data = GraphClassificationData.from_filepaths(["a.pt", "b.pt"], [0, 1]) # doctest: +SKIP

        """

        # enable passing in a string which loads all files in that folder as a list
        if isinstance(train_filepaths, str):
            train_filepaths = [os.path.join(train_filepaths, x) for x in os.listdir(train_filepaths)]
        if isinstance(valid_filepaths, str):
            valid_filepaths = [os.path.join(valid_filepaths, x) for x in os.listdir(valid_filepaths)]
        if isinstance(test_filepaths, str):
            test_filepaths = [os.path.join(test_filepaths, x) for x in os.listdir(test_filepaths)]

        train_ds = FilepathDataset(
            filepaths=train_filepaths,
            labels=train_labels,
            loader=loader,
            transform=train_transform,
        )
        valid_ds = (
            FilepathDataset(
                filepaths=valid_filepaths,
                labels=valid_labels,
                loader=loader,
                transform=valid_transform,
            ) if valid_filepaths is not None else None
        )

        test_ds = (
            FilepathDataset(
                filepaths=test_filepaths,
                labels=test_labels,
                loader=loader,
                transform=valid_transform,
            ) if test_filepaths is not None else None
        )

        return cls(
            train_ds=train_ds,
            valid_ds=valid_ds,
            test_ds=test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[Union[str, pathlib.Path]],
        train_transform: Optional[Callable] = None,
        valid_folder: Optional[Union[str, pathlib.Path]] = None,
        valid_transform: Optional[Callable] = None,
        test_folder: Optional[Union[str, pathlib.Path]] = None,
        loader: Callable = torch.load,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        **kwargs
    ):
        """
        Creates a GraphClassificationData object from folders of Graphs arranged in this way: ::

            train/dog/xxx.png
            train/dog/xxy.png
            train/dog/xxz.png
            train/cat/123.png
            train/cat/nsdf3.png
            train/cat/asd932.png

        Args:
            train_folder: Path to training folder.
            train_transform: Graph transform to use for training set.
            valid_folder: Path to validation folder.
            valid_transform: Graph transform to use for validation and test set.
            test_folder: Path to test folder.
            loader: A function to load an Graph given its path.
            batch_size: Batch size for data loading.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to None which equals the number of available CPU threads.

        Returns:
            GraphClassificationData: the constructed data module

        Examples:
            >>> img_data = GraphClassificationData.from_folders("train/") # doctest: +SKIP

        """
        train_ds = FlashDatasetFolder(train_folder, transform=train_transform, loader=loader)
        valid_ds = (
            FlashDatasetFolder(valid_folder, transform=valid_transform, loader=loader)
            if valid_folder is not None else None
        )

        test_ds = (
            FlashDatasetFolder(test_folder, transform=valid_transform, loader=loader)
            if test_folder is not None else None
        )

        datamodule = cls(
            train_ds=train_ds,
            valid_ds=valid_ds,
            test_ds=test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        datamodule.num_classes = len(train_ds.classes)
        datamodule.data_pipeline = GraphClassificationDataPipeline(
            train_transform=train_transform, valid_transform=valid_transform, loader=loader
        )
        return datamodule


class GraphClassificationDataPipeline(ClassificationDataPipeline):

    def __init__(
        self,
        train_transform: Optional[Callable] = None,
        valid_transform: Optional[Callable] = None,
        use_valid_transform: bool = True,
        loader: Callable = torch.load
    ):
        self._train_transform = train_transform
        self._valid_transform = valid_transform
        self._use_valid_transform = use_valid_transform
        self._loader = loader

    def before_collate(self, samples: Any) -> Any:
        if _contains_any_tensor(samples):
            return samples

        if isinstance(samples, str):
            samples = [samples]
        if isinstance(samples, (list, tuple)) and all(isinstance(p, str) for p in samples):
            outputs = []
            for sample in samples:
                output = self._loader(sample)
                transform = self._valid_transform if self._use_valid_transform else self._train_transform
                outputs.append(transform(output))
            return outputs
        raise MisconfigurationException("The samples should either be a tensor or a list of paths.")
