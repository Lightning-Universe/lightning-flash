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
from typing import Any, Collection, Dict, Optional, Sequence, Tuple, Type

import numpy as np
from torch import Tensor

from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_module import DataModule
from flash.core.data.io.input import Input
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE, _IMAGE_EXTRAS_TESTING, _IMAGE_TESTING, lazy_import
from flash.core.utilities.stages import RunningStage
from flash.core.utilities.types import INPUT_TRANSFORM_TYPE
from flash.image.segmentation.input import (
    SemanticSegmentationFiftyOneInput,
    SemanticSegmentationFilesInput,
    SemanticSegmentationFolderInput,
    SemanticSegmentationNumpyInput,
    SemanticSegmentationTensorInput,
)
from flash.image.segmentation.input_transform import SemanticSegmentationInputTransform
from flash.image.segmentation.viz import SegmentationMatplotlibVisualization

if _FIFTYONE_AVAILABLE:
    fo = lazy_import("fiftyone")
    SampleCollection = "fiftyone.core.collections.SampleCollection"
else:
    fo = None
    SampleCollection = object

# Skip doctests if requirements aren't available
__doctest_skip__ = []
if not _IMAGE_TESTING:
    __doctest_skip__ += [
        "SemanticSegmentationData",
        "SemanticSegmentationData.from_files",
        "SemanticSegmentationData.from_folders",
        "SemanticSegmentationData.from_numpy",
        "SemanticSegmentationData.from_tensors",
    ]
if not _IMAGE_EXTRAS_TESTING:
    __doctest_skip__ += ["SemanticSegmentationData.from_fiftyone"]


class SemanticSegmentationData(DataModule):
    """The ``SemanticSegmentationData`` class is a :class:`~flash.core.data.data_module.DataModule` with a set of
    classmethods for loading data for semantic segmentation."""

    input_transform_cls = SemanticSegmentationInputTransform

    @property
    def labels_map(self) -> Optional[Dict[int, Tuple[int, int, int]]]:
        return getattr(self.train_dataset, "labels_map", None)

    @classmethod
    def from_files(
        cls,
        train_files: Optional[Sequence[str]] = None,
        train_targets: Optional[Sequence[str]] = None,
        val_files: Optional[Sequence[str]] = None,
        val_targets: Optional[Sequence[str]] = None,
        test_files: Optional[Sequence[str]] = None,
        test_targets: Optional[Sequence[str]] = None,
        predict_files: Optional[Sequence[str]] = None,
        input_cls: Type[Input] = SemanticSegmentationFilesInput,
        num_classes: Optional[int] = None,
        labels_map: Dict[int, Tuple[int, int, int]] = None,
        transform: INPUT_TRANSFORM_TYPE = SemanticSegmentationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "SemanticSegmentationData":
        """Load the :class:`~flash.image.segmentation.data.SemanticSegmentationData` from lists of input files and
        corresponding lists of mask files.

        The supported file extensions are: ``.jpg``, ``.jpeg``, ``.png``, ``.ppm``, ``.bmp``, ``.pgm``, ``.tif``,
        ``.tiff``, ``.webp``, and ``.npy``.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_files: The list of image files to use when training.
            train_targets: The list of mask files to use when training.
            val_files: The list of image files to use when validating.
            val_targets: The list of mask files to use when validating.
            test_files: The list of image files to use when testing.
            test_targets: The list of mask files to use when testing.
            predict_files: The list of image files to use when predicting.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            num_classes: The number of segmentation classes.
            labels_map: An optional mapping from class to RGB tuple indicating the colour to use when visualizing masks.
                If not provided, a random mapping will be used.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.image.segmentation.data.SemanticSegmentationData`.

        Examples
        ________

        .. testsetup::

            >>> from PIL import Image
            >>> rand_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))
            >>> rand_mask= np.random.randint(0, 10, (64, 64), dtype="uint8")
            >>> _ = [rand_image.save(f"image_{i}.png") for i in range(1, 4)]
            >>> _ = [np.save(f"mask_{i}.npy", rand_mask) for i in range(1, 4)]
            >>> _ = [rand_image.save(f"predict_image_{i}.png") for i in range(1, 4)]

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.image import SemanticSegmentation, SemanticSegmentationData
            >>> datamodule = SemanticSegmentationData.from_files(
            ...     train_files=["image_1.png", "image_2.png", "image_3.png"],
            ...     train_targets=["mask_1.npy", "mask_2.npy", "mask_3.npy"],
            ...     predict_files=["predict_image_1.png", "predict_image_2.png", "predict_image_3.png"],
            ...     transform_kwargs=dict(image_size=(128, 128)),
            ...     num_classes=10,
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            10
            >>> model = SemanticSegmentation(backbone="resnet18", num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> import os
            >>> _ = [os.remove(f"image_{i}.png") for i in range(1, 4)]
            >>> _ = [os.remove(f"mask_{i}.npy") for i in range(1, 4)]
            >>> _ = [os.remove(f"predict_image_{i}.png") for i in range(1, 4)]
        """

        ds_kw = dict(
            num_classes=num_classes,
            labels_map=labels_map,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_files, train_targets, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_files, val_targets, **ds_kw),
            input_cls(RunningStage.TESTING, test_files, test_targets, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_files, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[str] = None,
        train_target_folder: Optional[str] = None,
        val_folder: Optional[str] = None,
        val_target_folder: Optional[str] = None,
        test_folder: Optional[str] = None,
        test_target_folder: Optional[str] = None,
        predict_folder: Optional[str] = None,
        input_cls: Type[Input] = SemanticSegmentationFolderInput,
        num_classes: Optional[int] = None,
        labels_map: Dict[int, Tuple[int, int, int]] = None,
        transform: INPUT_TRANSFORM_TYPE = SemanticSegmentationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "SemanticSegmentationData":
        """Load the :class:`~flash.image.segmentation.data.SemanticSegmentationData` from folders containing image
        files and folders containing mask files.

        The supported file extensions are: ``.jpg``, ``.jpeg``, ``.png``, ``.ppm``, ``.bmp``, ``.pgm``, ``.tif``,
        ``.tiff``, ``.webp``, and ``.npy``.
        For train, test, and validation data, the folders are expected to contain the images with a corresponding target
        folder which contains the mask in a file of the same name.
        For example, if your ``train_images`` folder (passed to the ``train_folder`` argument) looks like this:

        .. code-block::

            train_images
            ├── image_1.png
            ├── image_2.png
            ├── image_3.png
            ...

        your ``train_masks`` folder (passed to the ``train_target_folder`` argument) would need to look like this
        (although the file extensions could be different):

        .. code-block::

            train_masks
            ├── image_1.png
            ├── image_2.png
            ├── image_3.png
            ...

        For prediction, the folder is expected to contain the files for inference, like this:

        .. code-block::

            predict_folder
            ├── predict_image_1.png
            ├── predict_image_2.png
            ├── predict_image_3.png
            ...

        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_folder: The folder containing images to use when training.
            train_target_folder: The folder containing masks to use when training (files should have the same name as
                the files in the ``train_folder``).
            val_folder: The folder containing images to use when validating.
            val_target_folder: The folder containing masks to use when validating (files should have the same name as
                the files in the ``train_folder``).
            test_folder: The folder containing images to use when testing.
            test_target_folder: The folder containing masks to use when testing (files should have the same name as
                the files in the ``train_folder``).
            predict_folder: The folder containing images to use when predicting.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            num_classes: The number of segmentation classes.
            labels_map: An optional mapping from class to RGB tuple indicating the colour to use when visualizing masks.
                If not provided, a random mapping will be used.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
              :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.image.segmentation.data.SemanticSegmentationData`.

        Examples
        ________

        .. testsetup::

            >>> import os
            >>> from PIL import Image
            >>> rand_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))
            >>> rand_mask = Image.fromarray(np.random.randint(0, 10, (64, 64), dtype="uint8"))
            >>> os.makedirs("train_images", exist_ok=True)
            >>> os.makedirs("train_masks", exist_ok=True)
            >>> os.makedirs("predict_folder", exist_ok=True)
            >>> _ = [rand_image.save(os.path.join("train_images", f"image_{i}.png")) for i in range(1, 4)]
            >>> _ = [rand_mask.save(os.path.join("train_masks", f"image_{i}.png")) for i in range(1, 4)]
            >>> _ = [rand_image.save(os.path.join("predict_folder", f"predict_image_{i}.png")) for i in range(1, 4)]

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.image import SemanticSegmentation, SemanticSegmentationData
            >>> datamodule = SemanticSegmentationData.from_folders(
            ...     train_folder="train_images",
            ...     train_target_folder="train_masks",
            ...     predict_folder="predict_folder",
            ...     transform_kwargs=dict(image_size=(128, 128)),
            ...     num_classes=10,
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            10
            >>> model = SemanticSegmentation(backbone="resnet18", num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> import shutil
            >>> shutil.rmtree("train_images")
            >>> shutil.rmtree("train_masks")
            >>> shutil.rmtree("predict_folder")
        """

        ds_kw = dict(
            num_classes=num_classes,
            labels_map=labels_map,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_folder, train_target_folder, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_folder, val_target_folder, **ds_kw),
            input_cls(RunningStage.TESTING, test_folder, test_target_folder, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_folder, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_numpy(
        cls,
        train_data: Optional[Collection[np.ndarray]] = None,
        train_targets: Optional[Collection[np.ndarray]] = None,
        val_data: Optional[Collection[np.ndarray]] = None,
        val_targets: Optional[Collection[np.ndarray]] = None,
        test_data: Optional[Collection[np.ndarray]] = None,
        test_targets: Optional[Collection[np.ndarray]] = None,
        predict_data: Optional[Collection[np.ndarray]] = None,
        input_cls: Type[Input] = SemanticSegmentationNumpyInput,
        num_classes: Optional[int] = None,
        labels_map: Dict[int, Tuple[int, int, int]] = None,
        transform: INPUT_TRANSFORM_TYPE = SemanticSegmentationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "SemanticSegmentationData":
        """Load the :class:`~flash.image.segmentation.data.SemanticSegmentationData` from numpy arrays containing
        images (or lists of arrays) and corresponding numpy arrays containing masks (or lists of arrays).

        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_data: The numpy array or list of arrays containing images to use when training.
            train_targets: The numpy array or list of arrays containing masks to use when training.
            val_data: The numpy array or list of arrays containing images to use when validating.
            val_targets: The numpy array or list of arrays containing masks to use when validating.
            test_data: The numpy array or list of arrays containing images to use when testing.
            test_targets: The numpy array or list of arrays containing masks to use when testing.
            predict_data: The numpy array or list of arrays to use when predicting.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            num_classes: The number of segmentation classes.
            labels_map: An optional mapping from class to RGB tuple indicating the colour to use when visualizing masks.
                If not provided, a random mapping will be used.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
              :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.image.segmentation.data.SemanticSegmentationData`.

        Examples
        ________

        .. doctest::

            >>> import numpy as np
            >>> from flash import Trainer
            >>> from flash.image import SemanticSegmentation, SemanticSegmentationData
            >>> datamodule = SemanticSegmentationData.from_numpy(
            ...     train_data=[np.random.rand(3, 64, 64), np.random.rand(3, 64, 64), np.random.rand(3, 64, 64)],
            ...     train_targets=[
            ...         np.random.randint(0, 10, (1, 64, 64), dtype="uint8"),
            ...         np.random.randint(0, 10, (1, 64, 64), dtype="uint8"),
            ...         np.random.randint(0, 10, (1, 64, 64), dtype="uint8"),
            ...     ],
            ...     predict_data=[np.random.rand(3, 64, 64)],
            ...     transform_kwargs=dict(image_size=(128, 128)),
            ...     num_classes=10,
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            10
            >>> model = SemanticSegmentation(backbone="resnet18", num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...
        """

        ds_kw = dict(
            num_classes=num_classes,
            labels_map=labels_map,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_data, train_targets, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_data, val_targets, **ds_kw),
            input_cls(RunningStage.TESTING, test_data, test_targets, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_data, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_tensors(
        cls,
        train_data: Optional[Collection[Tensor]] = None,
        train_targets: Optional[Collection[Tensor]] = None,
        val_data: Optional[Collection[Tensor]] = None,
        val_targets: Optional[Collection[Tensor]] = None,
        test_data: Optional[Collection[Tensor]] = None,
        test_targets: Optional[Collection[Tensor]] = None,
        predict_data: Optional[Collection[Tensor]] = None,
        input_cls: Type[Input] = SemanticSegmentationTensorInput,
        num_classes: Optional[int] = None,
        labels_map: Dict[int, Tuple[int, int, int]] = None,
        transform: INPUT_TRANSFORM_TYPE = SemanticSegmentationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "SemanticSegmentationData":
        """Load the :class:`~flash.image.segmentation.data.SemanticSegmentationData` from torch tensors containing
        images (or lists of tensors) and corresponding torch tensors containing masks (or lists of tensors).

        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_data: The torch tensor or list of tensors containing images to use when training.
            train_targets: The torch tensor or list of tensors containing masks to use when training.
            val_data: The torch tensor or list of tensors containing images to use when validating.
            val_targets: The torch tensor or list of tensors containing masks to use when validating.
            test_data: The torch tensor or list of tensors containing images to use when testing.
            test_targets: The torch tensor or list of tensors containing masks to use when testing.
            predict_data: The torch tensor or list of tensors to use when predicting.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            num_classes: The number of segmentation classes.
            labels_map: An optional mapping from class to RGB tuple indicating the colour to use when visualizing masks.
                If not provided, a random mapping will be used.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
              :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.image.segmentation.data.SemanticSegmentationData`.

        Examples
        ________

        .. doctest::

            >>> import torch
            >>> from flash import Trainer
            >>> from flash.image import SemanticSegmentation, SemanticSegmentationData
            >>> datamodule = SemanticSegmentationData.from_tensors(
            ...     train_data=[torch.rand(3, 64, 64), torch.rand(3, 64, 64), torch.rand(3, 64, 64)],
            ...     train_targets=[
            ...         torch.randint(10, (1, 64, 64)),
            ...         torch.randint(10, (1, 64, 64)),
            ...         torch.randint(10, (1, 64, 64)),
            ...     ],
            ...     predict_data=[torch.rand(3, 64, 64)],
            ...     transform_kwargs=dict(image_size=(128, 128)),
            ...     num_classes=10,
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            10
            >>> model = SemanticSegmentation(backbone="resnet18", num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...
        """

        ds_kw = dict(
            num_classes=num_classes,
            labels_map=labels_map,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_data, train_targets, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_data, val_targets, **ds_kw),
            input_cls(RunningStage.TESTING, test_data, test_targets, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_data, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_fiftyone(
        cls,
        train_dataset: Optional[SampleCollection] = None,
        val_dataset: Optional[SampleCollection] = None,
        test_dataset: Optional[SampleCollection] = None,
        predict_dataset: Optional[SampleCollection] = None,
        input_cls: Type[Input] = SemanticSegmentationFiftyOneInput,
        num_classes: Optional[int] = None,
        labels_map: Dict[int, Tuple[int, int, int]] = None,
        transform: INPUT_TRANSFORM_TYPE = SemanticSegmentationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        label_field: str = "ground_truth",
        **data_module_kwargs: Any,
    ) -> "SemanticSegmentationData":
        """Load the :class:`~flash.image.segmentation.data.SemanticSegmentationData` from FiftyOne
        ``SampleCollection`` objects.

        The supported file extensions are: ``.jpg``, ``.jpeg``, ``.png``, ``.ppm``, ``.bmp``, ``.pgm``, ``.tif``,
        ``.tiff``, ``.webp``, and ``.npy``.
        Mask image file paths will be extracted from the ``label_field`` in the ``SampleCollection`` objects.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_dataset: The ``SampleCollection`` to use when training.
            val_dataset: The ``SampleCollection`` to use when validating.
            test_dataset: The ``SampleCollection`` to use when testing.
            predict_dataset: The ``SampleCollection`` to use when predicting.
            label_field: The field in the ``SampleCollection`` objects containing the targets.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            num_classes: The number of segmentation classes.
            labels_map: An optional mapping from class to RGB tuple indicating the colour to use when visualizing masks.
                If not provided, a random mapping will be used.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
              :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.image.segmentation.data.SemanticSegmentationData`.

        Examples
        ________

        .. testsetup::

            >>> from PIL import Image
            >>> rand_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))
            >>> _ = [rand_image.save(f"image_{i}.png") for i in range(1, 4)]
            >>> _ = [rand_image.save(f"predict_image_{i}.png") for i in range(1, 4)]

        .. doctest::

            >>> import numpy as np
            >>> import fiftyone as fo
            >>> from flash import Trainer
            >>> from flash.image import SemanticSegmentation, SemanticSegmentationData
            >>> train_dataset = fo.Dataset.from_images(
            ...     ["image_1.png", "image_2.png", "image_3.png"]
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            <BLANKLINE>
            ...
            >>> samples = [train_dataset[filepath] for filepath in train_dataset.values("filepath")]
            >>> for sample in samples:
            ...     sample["ground_truth"] = fo.Segmentation(mask=np.random.randint(0, 10, (64, 64), dtype="uint8"))
            ...     sample.save()
            ...
            >>> predict_dataset = fo.Dataset.from_images(
            ...     ["predict_image_1.png", "predict_image_2.png", "predict_image_3.png"]
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            <BLANKLINE>
            ...
            >>> datamodule = SemanticSegmentationData.from_fiftyone(
            ...     train_dataset=train_dataset,
            ...     predict_dataset=predict_dataset,
            ...     transform_kwargs=dict(image_size=(128, 128)),
            ...     num_classes=10,
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            10
            >>> model = SemanticSegmentation(backbone="resnet18", num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> import os
            >>> _ = [os.remove(f"image_{i}.png") for i in range(1, 4)]
            >>> _ = [os.remove(f"predict_image_{i}.png") for i in range(1, 4)]
        """

        return cls(
            input_cls(
                RunningStage.TRAINING,
                train_dataset,
                label_field=label_field,
                num_classes=num_classes,
                labels_map=labels_map,
            ),
            input_cls(
                RunningStage.VALIDATING,
                val_dataset,
                label_field=label_field,
                num_classes=num_classes,
                labels_map=labels_map,
            ),
            input_cls(
                RunningStage.TESTING,
                test_dataset,
                label_field=label_field,
                num_classes=num_classes,
                labels_map=labels_map,
            ),
            input_cls(RunningStage.PREDICTING, predict_dataset),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    def configure_data_fetcher(self) -> BaseDataFetcher:
        return SegmentationMatplotlibVisualization(labels_map=self.labels_map)

    def set_block_viz_window(self, value: bool) -> None:
        """Setter method to switch on/off matplotlib to pop up windows."""
        self.data_fetcher.block_viz_window = value
