import pathlib
import os

from pl_flash.vision.data.base import VisionData
from pl_flash.data import FlashDataModule
from typing import Any, Optional, Tuple, Union, Callable
import torch

__all__ = ["ImageSegmentationData"]

try:
    from torchvision.datasets.vision import StandardTransform
except ImportError:

    class StandardTransform:
        """Basic Drop-In transform interface for torchvision StandardTransform

        Args:
            transform: the transforms to apply to the images. Defaults to None.
            target_transform: the transforms to apply to the targets. Defaults to None.
        """

        def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):

            self.transform = transform
            self.target_transform = target_transform

        def __call__(self, inputs: Any, targets: Any) -> Tuple[Any, Any]:

            if self.transform is not None:
                inputs = self.transform(inputs)

            if self.target_transform is not None:
                targets = self.target_transform(targets)

            return inputs, targets


class FileSuffixDataset(torch.utils.data.Dataset):
    """Dataset loading images and masks from one folder where masks are named the same as images with a suffix

    Args:
        root_path: the path to the directory containing images and masks
        loader: the function to load the actual images
        suffix: the suffix that matches masks to images. Defaults to "_mask".
        transform: the transforms to apply to the image. Won't be used, if :attr:`transforms` is specified.
        target_transform: the transforms to apply to the target. Won't be used, if :attr:`transforms` is specified.
        transforms: the transforms to apply to the image and targets. If specified, overwrites :attr:`transform`
            and :attr`target_transform`

    """

    def __init__(
        self,
        root_path: Union[str, pathlib.Path],
        loader: Callable,
        suffix: str = "_mask",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:

        super().__init__()

        self.data = self.parse_dir(root_path, suffix)
        self.loader = loader

        # from torchvision/datasets/vision.py
        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can " "be passed as argument")

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

    @staticmethod
    def parse_dir(path: Union[str, pathlib.Path], suffix: str) -> tuple:
        """parses the directory to create a list of tuples each with a file for image and a file for mask

        Args:
            path: the path to the directory containing images and masks
            suffix: the suffix that matches masks to images.

        Returns:
            tuple: the sequence of data samples
        """
        path = str(path)

        data = []

        for item in [os.path.join(path, x) for x in os.listdir(path)]:
            if not os.path.isdir(item):
                continue

            file, ext = os.path.splitext(item)

            mask_file = file + suffix + os.path.extsep + ext

            if os.path.isfile(mask_file):
                data.append((item, mask_file))

        return tuple(data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads and returns a sample corresponding to the given index

        Args:
            index: the index to correspond the value

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: the loaded sample being a tuple of image and mask
        """
        data_file, mask_file = self.data[index]

        data = self.loader(data_file)
        mask = self.loader(mask_file)

        data = self.format_data(data).float()
        mask = self.format_mask(mask).long()

        if self.transforms is not None:
            data, mask = self.transforms(data, mask)

        return data, mask

    @staticmethod
    def format_data(data: torch.Tensor) -> torch.Tensor:
        """formats the data to ensure that we always have channels at front

        Args:
            data: the image tensor to format

        Returns:
            torch.Tensor: the formatted image tensor
        """
        if data.ndim == 2:
            return data[None]

        if data.ndim == 3:
            if data.size(0) in [1, 3, 4]:
                return data
            elif data.size(-1) in [1, 3, 4]:
                return data.permute(2, 0, 1)

        return data

    @staticmethod
    def format_mask(mask: torch.Tensor) -> torch.Tensor:
        """formats the mask to ensure it has no channels

        Args:
            mask: the mask to format correctly

        Returns:
            torch.Tensor: the formatted mask
        """
        if mask.ndim == 2:
            return mask

        if mask.ndim == 3:
            if mask.size(0) == 1:
                return mask[0]

            elif mask.size(-1) == 1:
                return mask[..., 0]

            else:
                return mask.argmax(0)

        return mask

    def __len__(self) -> int:
        return len(self.data)


class FilePathDirDataset(FileSuffixDataset):
    def __init__(
        self,
        image_path: Union[str, pathlib.Path],
        mask_path,
        loader: Callable,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        """A dataset loading image and mask from two different directories.
        The files must have the same name in both, the image and the mask directory.

        Args:
            image_path: the path to the image directory
            mask_path: the path to the mask directory
            loader: the function to load the actual image and mask (separately)
            transform: the transforms to apply to the image. Won't be used, if :attr:`transforms` is specified.
            target_transform: the transforms to apply to the target. Won't be used, if :attr:`transforms` is specified.
            transforms: the transforms to apply to the image and targets. If specified, overwrites :attr:`transform`
                and :attr`target_transform`


        """
        super().__init__(
            root_path=(image_path, mask_path),
            loader=loader,
            suffix="",
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )

    @staticmethod
    def parse_dir(path: Tuple[Union[str, pathlib.Path], Union[str, pathlib.Path]]) -> tuple:
        """parses the given directories to get all valid pairs of image and mask

        Args:
            path: the tuple consisting of image and mask path

        Returns:
            tuple: a sequence of all found samples each consisting of an image and a mask file
        """
        image_path, mask_path = [str(_path) for _path in path]

        data = []

        for file in os.listdir(image_path):
            image_file = os.path.join(image_path, file)
            mask_file = os.path.join(mask_path, file)

            if os.path.isfile(image_file) and os.path.isfile(mask_file):
                data.append((image_file, mask_file))

        return tuple(data)


class ImageSegmentationData(FlashDataModule, VisionData):
    """Data module for image segmentation tasks."""

    @classmethod
    def from_same_folder(
        cls,
        suffix: str = "_mask",
        train_dir: Optional[Union[str, pathlib.Path]] = None,
        train_transform: Optional[Callable] = None,
        train_target_transform: Optional[Callable] = None,
        train_transforms: Optional[Callable] = None,
        valid_dir: Optional[Union[str, pathlib.Path]] = None,
        valid_transform: Optional[Callable] = None,
        valid_target_transform: Optional[Callable] = None,
        valid_transforms: Optional[Callable] = None,
        test_dir: Optional[Union[str, pathlib.Path]] = None,
        loader: Optional[Callable] = None,
        batch_size: int = 64,
        num_workers: Optional[int] = None,
        **kwargs
    ) -> FlashDataModule:
        """Creates a Data Module from one folder with files like this (having a suffix of '_mask'):
        dir/a.png
        dir/a_mask.png
        dir/b.png
        dir/b_mask.png
        dir/c.jpg
        dir/c_mask.jpg
        ...

        Args:
            suffix: the actual suffix to distinguish masks and images. Defaults to "_mask".
            train_dir: the training directory. Defaults to None.
            train_transform: the transforms to apply to the images of the train dataset.
                Won't be used, if :attr:`train_transforms` is specified.
            train_target_transform: the transforms to apply to the targets of the train dataset.
                Won't be used, if :attr:`train_transforms` is specified.
            train_transforms: the transforms to apply to the image and targets of the train dataset.
                If specified, overwrites :attr:`train_transform` and :attr`train_target_transform`
            valid_dir: the validation data directory. Defaults to None.
            valid_transform: the transforms to apply to the images of the valid dataset.
                Won't be used, if :attr:`valid_transforms` is specified.
            valid_target_transform: the transforms to apply to the targets of the valid dataset.
                Won't be used, if :attr:`valid_transforms` is specified.
            valid_transforms: the transforms to apply to the image and targets of the valid dataset.
                If specified, overwrites :attr:`valid_transform` and :attr`valid_target_transform`
            test_dir: the testing data directory. Defaults to None.
            loader: function to load a single image (will be used for both, image and mask). Defaults to None.
            batch_size: the batch size to use for training. Defaults to 64.
            num_workers: the number of processes to use for parallel loading. Defaults to None.

        Returns:
            FlashDataModule: the created data module
        """

        if loader is None:
            loader = cls.load_file

        train_ds, valid_ds, test_ds = None, None, None

        if train_dir is not None:
            train_ds = FileSuffixDataset(
                root_path=train_dir,
                loader=loader,
                suffix=suffix,
                transform=train_transform,
                target_transform=train_target_transform,
                transforms=train_transforms,
            )

        if valid_dir is not None:
            valid_ds = FileSuffixDataset(
                root_path=valid_dir,
                loader=loader,
                suffix=suffix,
                transform=valid_transform,
                target_transform=valid_target_transform,
                transforms=valid_transforms,
            )

        if test_dir is not None:
            test_ds = FileSuffixDataset(
                root_path=test_dir,
                loader=loader,
                suffix=suffix,
                transform=valid_transform,
                target_transform=valid_target_transform,
                transforms=valid_transforms,
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
    def from_different_folders(
        cls,
        train_image_dir: Optional[Union[str, pathlib.Path]] = None,
        train_mask_dir: Optional[Union[str, pathlib.Path]] = None,
        train_transform: Optional[Callable] = None,
        train_target_transform: Optional[Callable] = None,
        train_transforms: Optional[Callable] = None,
        valid_image_dir: Optional[Union[str, pathlib.Path]] = None,
        valid_mask_dir: Optional[Union[str, pathlib.Path]] = None,
        valid_transform: Optional[Callable] = None,
        valid_target_transform: Optional[Callable] = None,
        valid_transforms: Optional[Callable] = None,
        test_image_dir: Optional[Union[str, pathlib.Path]] = None,
        test_mask_dir: Optional[Union[str, pathlib.Path]] = None,
        loader: Optional[Callable] = None,
        batch_size: int = 64,
        num_workers: Optional[int] = None,
        **kwargs
    ) -> FlashDataModule:
        """creates a DataModule from different folders for images and masks.
        The files in both directories must be named the same:

        image_dir/a.png
        image_dir/b.png
        image_dir/c.png
        ...

        mask_dir/a.png
        mask_dir/b.png
        mask_dir/c.png
        ...


        Args:
            train_image_dir: the image directory for the train dataset. Defaults to None.
            train_mask_dir: the mask directory for the train dataset. Defaults to None.
            train_transform: the transforms to apply to the images of the train dataset.
                Won't be used, if :attr:`train_transforms` is specified.
            train_target_transform: the transforms to apply to the targets of the train dataset.
                Won't be used, if :attr:`train_transforms` is specified.
            train_transforms: the transforms to apply to the image and targets of the train dataset.
                If specified, overwrites :attr:`train_transform` and :attr`train_target_transform`
            valid_image_dir: the image directory for the validation dataset. Defaults to None.
            valid_mask_dir: the mask directory for the validation dataset. Defaults to None.
            valid_transform: the transforms to apply to the images of the valid dataset.
                Won't be used, if :attr:`valid_transforms` is specified.
            valid_target_transform: the transforms to apply to the targets of the valid dataset.
                Won't be used, if :attr:`valid_transforms` is specified.
            valid_transforms: the transforms to apply to the image and targets of the valid dataset.
                If specified, overwrites :attr:`valid_transform` and :attr`valid_target_transform`
            test_image_dir: the image directory for test data. Defaults to None.
            test_mask_dir: the mask directory for test data. Defaults to None.
            loader: function to load a single image (will be applied to image and mask files). Defaults to None.
            batch_size: the batchsize to use for training. Defaults to 64.
            num_workers: the number of processes to use for parallel loading. Defaults to None.

        Raises:
            ValueError: If an image_dir is given but the corresponding mask_dir is None

        Returns:
            FlashDataModule: the created data module
        """

        if loader is None:
            loader = cls.load_file

        train_ds, valid_ds, test_ds = None, None, None

        if train_image_dir is not None:
            if train_mask_dir is None:
                raise ValueError("Cannot have a specified image dir without a mask dir for train dataset")

            train_ds = FilePathDirDataset(
                image_path=train_image_dir,
                mask_path=train_mask_dir,
                loader=loader,
                transform=train_transform,
                target_transform=train_target_transform,
                transforms=train_transforms,
            )

        if valid_image_dir is not None:
            if valid_mask_dir is None:
                raise ValueError("Cannot have a specified image dir without a mask dir for valid dataset")

            valid_ds = FilePathDirDataset(
                image_path=valid_image_dir,
                mask_path=valid_mask_dir,
                loader=loader,
                transform=valid_transform,
                target_transform=valid_target_transform,
                transforms=valid_transforms,
            )

        if test_image_dir is not None:
            if test_mask_dir is None:
                raise ValueError("Cannot have a specified image dir without a mask dir for test dataset")

            test_ds = FilePathDirDataset(
                image_path=test_image_dir,
                mask_path=test_mask_dir,
                loader=loader,
                transform=valid_transform,
                target_transform=valid_target_transform,
                transforms=valid_transforms,
            )

        return cls(
            train_ds=train_ds,
            valid_ds=valid_ds,
            test_ds=test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs
        )
