import os
import pathlib
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import IMG_EXTENSIONS, make_dataset

from flash.core.data.datamodule import DataModule


def _pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        with Image.open(f) as img:
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
        if self.has_labels:
            self.label_to_class_mapping = {v: k for k, v in enumerate(list(sorted(list(set(self.fnames)))))}

    @property
    def has_labels(self):
        return self.labels is not None

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, index: int) -> Tuple[Any, Optional[int]]:
        filename = self.fnames[index]
        img = self.loader(filename)
        label = None
        if self.has_labels:
            label = self.label_to_class_mapping[filename]
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
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root,
        loader,
        extensions=IMG_EXTENSIONS,
        transform=None,
        target_transform=None,
        is_valid_file=None,
        predict=False,
        img_paths=[],  # todo: dont pass mutable defaults
    ):
        super(FlashDatasetFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        self.predict = predict
        self.loader = loader
        self.extensions = extensions

        if not predict:
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
            self.samples = img_paths

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
        if self.predict:
            path = self.samples[index]
            sample = self.loader(path)
            return self.transform(sample), -1
        else:
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return sample, target

    def __len__(self):
        return len(self.samples)


class ImageClassificationData(DataModule):
    """Data module for image classification tasks."""

    default_train_transforms = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    default_valid_transforms = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    @classmethod
    def from_filepaths(
        cls,
        train_filepaths: Optional[Sequence[Union[str, pathlib.Path]]] = None,
        train_labels: Optional[Sequence] = None,
        train_transform: Optional[Callable] = default_train_transforms,
        valid_filepaths: Optional[Sequence[Union[str, pathlib.Path]]] = None,
        valid_labels: Optional[Sequence] = None,
        valid_transform: Optional[Callable] = default_valid_transforms,
        test_filepaths: Optional[Sequence[Union[str, pathlib.Path]]] = None,
        test_labels: Optional[Sequence] = None,
        loader: Callable = _pil_loader,
        batch_size: int = 64,
        num_workers: Optional[int] = None,
        **kwargs
    ):
        """Creates a ImageClassificationData object from lists of image filepaths and labels

        Args:
            train_filepaths: sequence of file paths for training dataset. Defaults to None.
            train_labels: sequence of labels for training dataset. Defaults to None.
            train_transform: transforms for training dataset. Defaults to None.
            valid_filepaths: sequence of file paths for validation dataset. Defaults to None.
            valid_labels: sequence of labels for validation dataset. Defaults to None.
            valid_transform: transforms for validation and testing dataset. Defaults to None.
            test_filepaths: sequence of file paths for test dataset. Defaults to None.
            test_labels: sequence of labels for test dataset. Defaults to None.
            loader: function to load an image file. Defaults to None.
            batch_size: the batchsize to use for parallel loading. Defaults to 64.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to None which equals the number of available CPU threads.

        Returns:
            ImageClassificationData: The constructed data module.

        Examples:
            >>> img_data = ImageClassificationData.from_filepaths(["a.png", "b.png"], [0, 1]) # doctest: +SKIP

        """
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
        train_transform: Optional[Callable] = default_train_transforms,
        valid_folder: Optional[Union[str, pathlib.Path]] = None,
        valid_transform: Optional[Callable] = default_valid_transforms,
        test_folder: Optional[Union[str, pathlib.Path]] = None,
        loader: Callable = _pil_loader,
        batch_size: int = 64,
        num_workers: Optional[int] = None,
        **kwargs
    ):
        """
        Creates a ImageClassificationData object from folders of images arranged in this way: ::

            train/dog/xxx.png
            train/dog/xxy.png
            train/dog/xxz.png
            train/cat/123.png
            train/cat/nsdf3.png
            train/cat/asd932_.png

        Args:
            train_folder: Path to training folder.
            train_transform: Image transform to use for training set.
            valid_folder: Path to validation folder.
            valid_transform: Image transform to use for validation and test set.
            test_folder: Path to test folder.
            loader: A function to load an image given its path.
            batch_size: Batch size for data loading.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to None which equals the number of available CPU threads.

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

        return datamodule
