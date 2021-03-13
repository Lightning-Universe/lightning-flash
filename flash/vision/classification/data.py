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
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image, UnidentifiedImageError
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torchvision import transforms as T
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import has_file_allowed_extension, IMG_EXTENSIONS, make_dataset

from flash.core.classification import ClassificationDataPipeline
from flash.core.data.datamodule import DataModule
from flash.core.data.utils import _contains_any_tensor


def _pil_loader(path) -> Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f, Image.open(f) as img:
        return img.convert("RGB")


class FilepathDataset(torch.utils.data.Dataset):
    """Dataset that takes in filepaths and labels."""

    def __init__(
        self,
        filepaths: Optional[Sequence[Union[str, pathlib.Path]]],
        labels: Optional[Sequence],
        loader: Callable,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            filepaths: file paths to load with :attr:`loader`
            labels: the labels corresponding to the :attr:`filepaths`.
                Each unique value will get a class index by sorting them.
            loader: the function to load an image from a given file path
            transform: the transforms to apply to the loaded images
        """
        self.fnames = filepaths or []
        self.labels = labels or []
        self.transform = transform
        self.loader = loader
        if not self.has_dict_labels and self.has_labels:
            self.label_to_class_mapping = dict(map(reversed, enumerate(sorted(set(self.labels)))))

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
        img = self.loader(filename)
        if self.transform is not None:
            img = self.transform(img)
        label = None
        if self.has_dict_labels:
            name = os.path.splitext(filename)[0]
            name = os.path.basename(name)
            label = self.labels[name]

        elif self.has_labels:
            label = self.labels[index]
            label = self.label_to_class_mapping[label]
        return img, label


class FlashDatasetFolder(VisionDataset):
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
            E.g, ``transforms.RandomCrop`` for images.
        target_transform: A function/transform that takes
            in the target and transforms it.
        is_valid_file: A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
        with_targets: Whether to include targets
        img_paths: List of image paths to load. Only used when ``with_targets=False``

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root: str,
        loader: Callable,
        extensions: Tuple[str] = IMG_EXTENSIONS,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable] = None,
        with_targets: bool = True,
        img_paths: Optional[List[str]] = None,
    ):
        super(FlashDatasetFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        self.loader = loader
        self.extensions = extensions
        self.with_targets = with_targets

        if with_targets:
            classes, class_to_idx = self._find_classes(self.root)
            samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)

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
            if not img_paths:
                raise MisconfigurationException(
                    "`FlashDatasetFolder(with_target=False)` but no `img_paths` were provided"
                )
            self.samples = img_paths

    @staticmethod
    def _find_classes(folder: str):
        """
        Finds the class folders in a dataset.

        Args:
            folder: Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(folder) if d.is_dir()]
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


_DEFAULT_TRAIN_TRANSFORMS = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_DEFAULT_VALID_TRANSFORMS = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# todo: torch.nn.modules.module.ModuleAttributeError: 'Resize' object has no attribute '_forward_hooks'
# Find better fix and raised issue on torchvision.
_DEFAULT_VALID_TRANSFORMS.transforms[0]._forward_hooks = {}


class ImageClassificationDataPipeline(ClassificationDataPipeline):

    def __init__(
        self,
        train_transform: Optional[Callable] = _DEFAULT_TRAIN_TRANSFORMS,
        valid_transform: Optional[Callable] = _DEFAULT_VALID_TRANSFORMS,
        use_valid_transform: bool = True,
        loader: Callable = _pil_loader
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
                try:
                    output = self._loader(sample)
                    transform = self._valid_transform if self._use_valid_transform else self._train_transform
                    outputs.append(transform(output))
                except UnidentifiedImageError:
                    print(f'Skipping: could not read file {sample}')

            return outputs
        raise MisconfigurationException("The samples should either be a tensor or a list of paths.")


class ImageClassificationData(DataModule):
    """Data module for image classification tasks."""

    @classmethod
    def from_filepaths(
        cls,
        train_filepaths: Union[str, Optional[Sequence[Union[str, pathlib.Path]]]] = None,
        train_labels: Optional[Sequence] = None,
        train_transform: Optional[Callable] = _DEFAULT_TRAIN_TRANSFORMS,
        valid_split: Union[None, float] = None,
        valid_filepaths: Union[str, Optional[Sequence[Union[str, pathlib.Path]]]] = None,
        valid_labels: Optional[Sequence] = None,
        valid_transform: Optional[Callable] = _DEFAULT_VALID_TRANSFORMS,
        test_filepaths: Union[str, Optional[Sequence[Union[str, pathlib.Path]]]] = None,
        test_labels: Optional[Sequence] = None,
        loader: Callable = _pil_loader,
        batch_size: int = 64,
        num_workers: Optional[int] = None,
        seed: int = 1234,
        **kwargs,
    ):
        """Creates a ImageClassificationData object from lists of image filepaths and labels

        Args:
            train_filepaths: string or sequence of file paths for training dataset. Defaults to ``None``.
            train_labels: sequence of labels for training dataset. Defaults to ``None``.
            train_transform: transforms for training dataset. Defaults to ``None``.
            valid_split: if not None, generates val split from train dataloader using this value.
            valid_filepaths: string or sequence of file paths for validation dataset. Defaults to ``None``.
            valid_labels: sequence of labels for validation dataset. Defaults to ``None``.
            valid_transform: transforms for validation and testing dataset. Defaults to ``None``.
            test_filepaths: string or sequence of file paths for test dataset. Defaults to ``None``.
            test_labels: sequence of labels for test dataset. Defaults to ``None``.
            loader: function to load an image file. Defaults to ``None``.
            batch_size: the batchsize to use for parallel loading. Defaults to ``64``.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to ``None`` which equals the number of available CPU threads.
            seed: Used for the train/val splits when valid_split is not None

        Returns:
            ImageClassificationData: The constructed data module.

        Examples:
            >>> img_data = ImageClassificationData.from_filepaths(["a.png", "b.png"], [0, 1]) # doctest: +SKIP

        Example when labels are in .csv file::

            train_labels = labels_from_categorical_csv('path/to/train.csv', 'my_id')
            valid_labels = labels_from_categorical_csv(path/to/valid.csv', 'my_id')
            test_labels = labels_from_categorical_csv(path/to/tests.csv', 'my_id')

            data = ImageClassificationData.from_filepaths(
                batch_size=2,
                train_filepaths='path/to/train',
                train_labels=train_labels,
                valid_filepaths='path/to/valid',
                valid_labels=valid_labels,
                test_filepaths='path/to/test',
                test_labels=test_labels,
            )

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

        if valid_split:
            full_length = len(train_ds)
            train_split = int((1.0 - valid_split) * full_length)
            valid_split = full_length - train_split
            train_ds, valid_ds = torch.utils.data.random_split(
                train_ds, [train_split, valid_split], generator=torch.Generator().manual_seed(seed)
            )
        else:
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
        train_transform: Optional[Callable] = _DEFAULT_TRAIN_TRANSFORMS,
        valid_folder: Optional[Union[str, pathlib.Path]] = None,
        valid_transform: Optional[Callable] = _DEFAULT_VALID_TRANSFORMS,
        test_folder: Optional[Union[str, pathlib.Path]] = None,
        loader: Callable = _pil_loader,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        **kwargs,
    ):
        """
        Creates a ImageClassificationData object from folders of images arranged in this way: ::

            train/dog/xxx.png
            train/dog/xxy.png
            train/dog/xxz.png
            train/cat/123.png
            train/cat/nsdf3.png
            train/cat/asd932.png

        Args:
            train_folder: Path to training folder.
            train_transform: Image transform to use for training set.
            valid_folder: Path to validation folder.
            valid_transform: Image transform to use for validation and test set.
            test_folder: Path to test folder.
            loader: A function to load an image given its path.
            batch_size: Batch size for data loading.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to ``None`` which equals the number of available CPU threads.

        Returns:
            ImageClassificationData: the constructed data module

        Examples:
            >>> img_data = ImageClassificationData.from_folders("train/") # doctest: +SKIP

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
        datamodule.data_pipeline = ImageClassificationDataPipeline(
            train_transform=train_transform, valid_transform=valid_transform, loader=loader
        )
        return datamodule

    @classmethod
    def from_folder(
        cls,
        folder: Union[str, pathlib.Path],
        transform: Optional[Callable] = _DEFAULT_VALID_TRANSFORMS,
        loader: Callable = _pil_loader,
        batch_size: int = 64,
        num_workers: Optional[int] = None,
        **kwargs,
    ):
        """
        Creates a ImageClassificationData object from folders of images arranged in this way: ::

            folder/dog_xxx.png
            folder/dog_xxy.png
            folder/dog_xxz.png
            folder/cat_123.png
            folder/cat_nsdf3.png
            folder/cat_asd932_.png

        Args:
            folder: Path to the data folder.
            transform: Image transform to apply to the data.
            loader: A function to load an image given its path.
            batch_size: Batch size for data loading.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to None which equals the number of available CPU threads.

        Returns:
            ImageClassificationData: the constructed data module

        Examples:
            >>> img_data = ImageClassificationData.from_folder("folder/") # doctest: +SKIP

        """
        if not os.path.isdir(folder):
            raise MisconfigurationException("folder should be a directory")

        filenames = os.listdir(folder)

        if any(not has_file_allowed_extension(f, IMG_EXTENSIONS) for f in filenames):
            raise MisconfigurationException(
                "No images with allowed extensions {IMG_EXTENSIONS} where found in {folder}"
            )

        test_ds = (
            FlashDatasetFolder(
                folder,
                transform=transform,
                loader=loader,
                with_targets=False,
                img_paths=[os.path.join(folder, f) for f in filenames]
            )
        )

        datamodule = cls(
            test_ds=test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        datamodule.data_pipeline = ImageClassificationDataPipeline(valid_transform=transform, loader=loader)
        return datamodule

    @staticmethod
    def default_pipeline() -> ImageClassificationDataPipeline:
        return ImageClassificationDataPipeline(
            train_transform=_DEFAULT_TRAIN_TRANSFORMS, valid_transform=_DEFAULT_VALID_TRANSFORMS, loader=_pil_loader
        )
