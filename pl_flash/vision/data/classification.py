import pathlib
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset

from pl_flash.vision.data.base import VisionData
from pl_flash.data import FlashDataModule

__all__ = ["ImageClassificationData"]


class FilepathDataset(Dataset):
    """Dataset that takes in filepaths and labels.

    Args:
        filepaths: file paths to load with :attr:`loader`
        labels: the labels corresponding to the :attr:`filepaths`. Each unique value will get a class index by sorting them.
        loader: the function to load an image from a given file path
        transform: the transforms to apply to the loaded images
    """

    def __init__(
        self,
        filepaths: Sequence[Union[str, pathlib.Path]],
        labels: Sequence,
        loader: Callable,
        transform: Optional[Callable] = None,
    ) -> None:
        self.fnames = filepaths
        self.labels = labels
        self.transform = transform
        self.loader = loader
        self.label_to_class_mapping = {v: k for k, v in enumerate(list(sorted(list(set(self.labels)))))}

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Union[int, torch.Tensor]]:
        img = self.loader(self.fnames[index])
        if self.transform:
            img = self.transform(img)
        return img, self.label_to_class_mapping[self.labels[index]]


class ImageClassificationData(FlashDataModule, VisionData):
    """Data module for image classification tasks."""

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
        loader: Optional[Callable] = None,
        batch_size: int = 64,
        num_workers: Optional[int] = None,
        **kwargs
    ) -> FlashDataModule:
        """Creates a ImageClassificationData object from lists of image filepaths and labels

        Args:
            train_filepaths: sequence of file paths for training dataset. Defaults to None.
            train_labels: sequence of labels for training dataset. Defaults to None.
            train_transform: transforms for training dataset. Defaults to None.
            valid_filepaths: sequence of file paths for validation dataset.. Defaults to None.
            valid_labels: sequence of labels for validation dataset. Defaults to None.
            valid_transform: transforms for validation and testing dataset.. Defaults to None.
            test_filepaths: sequence of file paths for test dataset.. Defaults to None.
            test_labels: sequence of labels for test dataset. Defaults to None.
            loader: function to load an image file. Defaults to None.
            batch_size: the batchsize to use for parallel loading. Defaults to 64.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to None which equals the number of available CPU threads.

        Returns:
            FlashDataModule: the constructed data module

        Examples:
            >>> img_data = ImageClassificationData.from_filepaths(["a.png", "b.png"], [0, 1]) # doctest: +SKIP
        """

        if loader is None:
            loader = cls.load_file

        train_ds = FilepathDataset(
            filepaths=train_filepaths, labels=train_labels, loader=loader, transform=train_transform
        )
        valid_ds = (
            FilepathDataset(filepaths=valid_filepaths, labels=valid_labels, loader=loader, transform=valid_transform)
            if valid_filepaths is not None
            else None
        )

        test_ds = (
            FilepathDataset(filepaths=test_filepaths, labels=test_labels, loader=loader, transform=valid_transform)
            if test_filepaths is not None
            else None
        )

        return cls(
            train_ds=train_ds,
            valid_ds=valid_ds,
            test_ds=test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs
        )

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[Union[str, pathlib.Path]],
        train_transform: Optional[Callable] = None,
        valid_folder: Optional[Union[str, pathlib.Path]] = None,
        valid_transform: Optional[Callable] = None,
        test_folder: Optional[Union[str, pathlib.Path]] = None,
        loader: Optional[Callable] = None,
        batch_size: int = 64,
        num_workers: Optional[int] = None,
        **kwargs
    ) -> FlashDataModule:
        """
        Creates a ImageClassificationData object from folders of images arranged in this way:

        train/dog/xxx.png
        train/dog/xxy.png
        train/dog/xxz.png

        train/cat/123.png
        train/cat/nsdf3.png
        train/cat/asd932{_}.png

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
            FlashDataModule: the constructed data module

        Examples::
            >>> img_data = ImageClassificationData.from_folders("train/") # doctest: +SKIP

        """
        try:
            from torchvision.datasets import ImageFolder

        except ImportError as e:
            raise ImportError(
                "Torchvision is not installed please install it following the guides on"
                + "https://pytorch.org/get-started/locally/"
            ) from e

        if loader is None:
            loader = cls.load_file

        train_ds = ImageFolder(train_folder, transform=train_transform, loader=loader)
        valid_ds = (
            ImageFolder(valid_folder, transform=valid_transform, loader=loader) if valid_folder is not None else None
        )

        test_ds = (
            ImageFolder(test_folder, transform=valid_transform, loader=loader) if test_folder is not None else None
        )

        return cls(
            train_ds=train_ds,
            valid_ds=valid_ds,
            test_ds=test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs
        )

    @property
    def default_train_transforms(self) -> Callable:
        try:
            from torchvision import transforms as T

        except ImportError as e:
            raise ImportError(
                "Torchvision is not installed please install it following the guides on"
                + "https://pytorch.org/get-started/locally/"
            ) from e
        train_transforms = T.Compose(
            [
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                # imagenet statistics
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        return train_transforms

    @property
    def default_validation_test_transforms(self) -> Callable:
        try:
            from torchvision import transforms as T

        except ImportError as e:
            raise ImportError(
                "Torchvision is not installed please install it following the guides on"
                + "https://pytorch.org/get-started/locally/"
            ) from e

        valid_transforms = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                # imagenet statistics
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        return valid_transforms
