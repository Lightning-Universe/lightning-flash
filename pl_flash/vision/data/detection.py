from typing import Callable, Optional
from pl_flash.data import FlashDataModule
from pl_flash.vision.data.base import VisionData

__all__ =["ImageDetectionData"]


class ImageDetectionData(FlashDataModule, VisionData):
    """Detection Data Module
    """
    @classmethod
    def from_coco_format(
        cls,
        train_dir: Optional[str] = None,
        train_annotation_file: Optional[str] = None,
        train_transform: Optional[Callable] = None,
        train_target_transform: Optional[Callable] = None,
        train_transforms: Optional[Callable] = None,
        valid_dir: Optional[str] = None,
        valid_annotation_file: Optional[str] = None,
        valid_transform: Optional[Callable] = None,
        valid_target_transform: Optional[Callable] = None,
        valid_transforms: Optional[Callable] = None,
        test_dir: Optional[str] = None,
        test_annotation_file: Optional[str] = None,
        batch_size: int = 64,
        num_workers: Optional[int] = None,
        **kwargs
    ) -> FlashDataModule:
        """creates a DataModule based on the Coco Format

        Args:
            train_dir: training root directory containing the images. Defaults to None.
            train_annotation_file: path to training json annotation file. Defaults to None.
            train_transform: transforms fo apply on the images of the trainset. Defaults to None.
            train_target_transform: transforms to apply to the targets of the trainset. Defaults to None.
            train_transforms: transforms to apply to both, images and targets of the trainset. Defaults to None.
            valid_dir: validation root directory contaning the images. Defaults to None.
            valid_annotation_file: path to validation json annotation file. Defaults to None.
            valid_transform: transforms to apply on the images of the validation dataset. Defaults to None.
            valid_target_transform: transforms to apply on the targets of the validation dataset. Defaults to None.
            valid_transforms: transforms to apply on both, the images and the targets of the validation dataset.
                Defaults to None.
            test_dir: the test root directory containing the images. Defaults to None.
            test_annotation_file: path to test json annotation file. Defaults to None.
            batch_size: the batchsize to use for training. Defaults to 64.
            num_workers: the number of workers to use for parallel loading. Defaults to None.

        Raises:
            ImportError: torchvision or pycocotools are not available

        Returns:
            FlashDataModule: [description]
        """

        try:
            import torchvision.datasets

        except ImportError as e:
            raise ImportError(
                "Torchvision is not installed please install it following the guides on"
                + "https://pytorch.org/get-started/locally/"
            ) from e

        try:
            import pycocotools
        except ImportError as e:
            raise ImportError("pycocotools are not installed. Pleas install it with `pip install pycocotools`") from e

        train_ds, valid_ds, test_ds = None, None, None

        if train_dir is not None:
            if train_annotation_file is None:
                raise ValueError("Cannot have a missing annotation file for a specified image directory")

            train_ds = torchvision.datasets.CocoDetection(
                root=train_dir,
                annFile=train_annotation_file,
                transform=train_transform,
                target_transform=train_target_transform,
                transforms=train_transforms,
            )

        if valid_dir is not None:
            if valid_annotation_file is None:
                raise ValueError("Cannot have a missing annotation file for a specified image directory")

            valid_ds = torchvision.datasets.CocoDetection(
                root=valid_dir,
                annFile=valid_annotation_file,
                transform=valid_transform,
                target_transform=valid_target_transform,
                transforms=valid_transforms,
            )

        if test_dir is not None:
            if test_annotation_file is None:
                raise ValueError("Cannot have a missing annotation file for a specified image directory")

            test_ds = torchvision.datasets.CocoDetection(
                root=test_dir,
                annFile=test_annotation_file,
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
