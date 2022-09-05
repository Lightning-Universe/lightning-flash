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
from typing import Any, Callable, Collection, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset

from flash.core.data.base_viz import BaseVisualization
from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_module import DataModule, DatasetInput
from flash.core.data.io.input import DataKeys, Input
from flash.core.data.io.input_transform import INPUT_TRANSFORM_TYPE
from flash.core.data.utilities.classification import TargetFormatter
from flash.core.data.utilities.paths import PATH_TYPE
from flash.core.integrations.labelstudio.input import _parse_labelstudio_arguments, LabelStudioImageClassificationInput
from flash.core.utilities.imports import (
    _FIFTYONE_AVAILABLE,
    _IMAGE_EXTRAS_TESTING,
    _IMAGE_TESTING,
    _MATPLOTLIB_AVAILABLE,
    Image,
    requires,
)
from flash.core.utilities.stages import RunningStage
from flash.image.classification.input import (
    ImageClassificationCSVInput,
    ImageClassificationDataFrameInput,
    ImageClassificationFiftyOneInput,
    ImageClassificationFilesInput,
    ImageClassificationFolderInput,
    ImageClassificationImageInput,
    ImageClassificationNumpyInput,
    ImageClassificationTensorInput,
)
from flash.image.classification.input_transform import ImageClassificationInputTransform

if _FIFTYONE_AVAILABLE:
    SampleCollection = "fiftyone.core.collections.SampleCollection"
else:
    SampleCollection = None

if _MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt
else:
    plt = None

# Skip doctests if requirements aren't available
__doctest_skip__ = []
if not _IMAGE_TESTING:
    __doctest_skip__ += [
        "ImageClassificationData",
        "ImageClassificationData.from_files",
        "ImageClassificationData.from_folders",
        "ImageClassificationData.from_numpy",
        "ImageClassificationData.from_images",
        "ImageClassificationData.from_tensors",
        "ImageClassificationData.from_data_frame",
        "ImageClassificationData.from_csv",
    ]
if not _IMAGE_EXTRAS_TESTING:
    __doctest_skip__ += ["ImageClassificationData.from_fiftyone"]


class ImageClassificationData(DataModule):
    """The ``ImageClassificationData`` class is a :class:`~flash.core.data.data_module.DataModule` with a set of
    classmethods for loading data for image classification."""

    input_transform_cls = ImageClassificationInputTransform

    @classmethod
    def from_files(
        cls,
        train_files: Optional[Sequence[str]] = None,
        train_targets: Optional[Sequence[Any]] = None,
        val_files: Optional[Sequence[str]] = None,
        val_targets: Optional[Sequence[Any]] = None,
        test_files: Optional[Sequence[str]] = None,
        test_targets: Optional[Sequence[Any]] = None,
        predict_files: Optional[Sequence[str]] = None,
        target_formatter: Optional[TargetFormatter] = None,
        input_cls: Type[Input] = ImageClassificationFilesInput,
        transform: INPUT_TRANSFORM_TYPE = ImageClassificationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "ImageClassificationData":
        """Load the :class:`~flash.image.classification.data.ImageClassificationData` from lists of files and
        corresponding lists of targets.

        The supported file extensions are: ``.jpg``, ``.jpeg``, ``.png``, ``.ppm``, ``.bmp``, ``.pgm``, ``.tif``,
        ``.tiff``, ``.webp``, and ``.npy``.
        The targets can be in any of our
        :ref:`supported classification target formats <formatting_classification_targets>`.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_files: The list of image files to use when training.
            train_targets: The list of targets to use when training.
            val_files: The list of image files to use when validating.
            val_targets: The list of targets to use when validating.
            test_files: The list of image files to use when testing.
            test_targets: The list of targets to use when testing.
            predict_files: The list of image files to use when predicting.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.image.classification.data.ImageClassificationData`.

        Examples
        ________

        .. testsetup::

            >>> from PIL import Image
            >>> rand_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))
            >>> _ = [rand_image.save(f"image_{i}.png") for i in range(1, 4)]
            >>> _ = [rand_image.save(f"predict_image_{i}.png") for i in range(1, 4)]

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.image import ImageClassifier, ImageClassificationData
            >>> datamodule = ImageClassificationData.from_files(
            ...     train_files=["image_1.png", "image_2.png", "image_3.png"],
            ...     train_targets=["cat", "dog", "cat"],
            ...     predict_files=["predict_image_1.png", "predict_image_2.png", "predict_image_3.png"],
            ...     transform_kwargs=dict(image_size=(128, 128)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['cat', 'dog']
            >>> model = ImageClassifier(backbone="resnet18", num_classes=datamodule.num_classes)
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
        ds_kw = dict(
            target_formatter=target_formatter,
        )

        train_input = input_cls(RunningStage.TRAINING, train_files, train_targets, **ds_kw)
        ds_kw["target_formatter"] = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
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
        val_folder: Optional[str] = None,
        test_folder: Optional[str] = None,
        predict_folder: Optional[str] = None,
        target_formatter: Optional[TargetFormatter] = None,
        input_cls: Type[Input] = ImageClassificationFolderInput,
        transform: INPUT_TRANSFORM_TYPE = ImageClassificationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "ImageClassificationData":
        """Load the :class:`~flash.image.classification.data.ImageClassificationData` from folders containing
        images.

        The supported file extensions are: ``.jpg``, ``.jpeg``, ``.png``, ``.ppm``, ``.bmp``, ``.pgm``, ``.tif``,
        ``.tiff``, ``.webp``, and ``.npy``.
        For train, test, and validation data, the folders are expected to contain a sub-folder for each class.
        Here's the required structure:

        .. code-block::

            train_folder
            ├── cat
            │   ├── image_1.png
            │   ├── image_3.png
            │   ...
            └── dog
                ├── image_2.png
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
            val_folder: The folder containing images to use when validating.
            test_folder: The folder containing images to use when testing.
            predict_folder: The folder containing images to use when predicting.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.image.classification.data.ImageClassificationData`.

        Examples
        ________

        .. testsetup::

            >>> import os
            >>> from PIL import Image
            >>> rand_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))
            >>> os.makedirs(os.path.join("train_folder", "cat"), exist_ok=True)
            >>> os.makedirs(os.path.join("train_folder", "dog"), exist_ok=True)
            >>> os.makedirs("predict_folder", exist_ok=True)
            >>> rand_image.save(os.path.join("train_folder", "cat", "image_1.png"))
            >>> rand_image.save(os.path.join("train_folder", "dog", "image_2.png"))
            >>> rand_image.save(os.path.join("train_folder", "cat", "image_3.png"))
            >>> _ = [rand_image.save(os.path.join("predict_folder", f"predict_image_{i}.png")) for i in range(1, 4)]

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.image import ImageClassifier, ImageClassificationData
            >>> datamodule = ImageClassificationData.from_folders(
            ...     train_folder="train_folder",
            ...     predict_folder="predict_folder",
            ...     transform_kwargs=dict(image_size=(128, 128)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['cat', 'dog']
            >>> model = ImageClassifier(backbone="resnet18", num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> import shutil
            >>> shutil.rmtree("train_folder")
            >>> shutil.rmtree("predict_folder")
        """
        ds_kw = dict(
            target_formatter=target_formatter,
        )

        train_input = input_cls(RunningStage.TRAINING, train_folder, **ds_kw)
        ds_kw["target_formatter"] = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_folder, **ds_kw),
            input_cls(RunningStage.TESTING, test_folder, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_folder, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_numpy(
        cls,
        train_data: Optional[Collection[np.ndarray]] = None,
        train_targets: Optional[Collection[Any]] = None,
        val_data: Optional[Collection[np.ndarray]] = None,
        val_targets: Optional[Sequence[Any]] = None,
        test_data: Optional[Collection[np.ndarray]] = None,
        test_targets: Optional[Sequence[Any]] = None,
        predict_data: Optional[Collection[np.ndarray]] = None,
        target_formatter: Optional[TargetFormatter] = None,
        input_cls: Type[Input] = ImageClassificationNumpyInput,
        transform: INPUT_TRANSFORM_TYPE = ImageClassificationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "ImageClassificationData":
        """Load the :class:`~flash.image.classification.data.ImageClassificationData` from numpy arrays (or lists
        of arrays) and corresponding lists of targets.

        The targets can be in any of our
        :ref:`supported classification target formats <formatting_classification_targets>`.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_data: The numpy array or list of arrays to use when training.
            train_targets: The list of targets to use when training.
            val_data: The numpy array or list of arrays to use when validating.
            val_targets: The list of targets to use when validating.
            test_data: The numpy array or list of arrays to use when testing.
            test_targets: The list of targets to use when testing.
            predict_data: The numpy array or list of arrays to use when predicting.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.image.classification.data.ImageClassificationData`.

        Examples
        ________

        .. doctest::

            >>> import numpy as np
            >>> from flash import Trainer
            >>> from flash.image import ImageClassifier, ImageClassificationData
            >>> datamodule = ImageClassificationData.from_numpy(
            ...     train_data=[np.random.rand(3, 64, 64), np.random.rand(3, 64, 64), np.random.rand(3, 64, 64)],
            ...     train_targets=["cat", "dog", "cat"],
            ...     predict_data=[np.random.rand(3, 64, 64)],
            ...     transform_kwargs=dict(image_size=(128, 128)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['cat', 'dog']
            >>> model = ImageClassifier(backbone="resnet18", num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...
        """
        ds_kw = dict(
            target_formatter=target_formatter,
        )

        train_input = input_cls(RunningStage.TRAINING, train_data, train_targets, **ds_kw)
        ds_kw["target_formatter"] = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_data, val_targets, **ds_kw),
            input_cls(RunningStage.TESTING, test_data, test_targets, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_data, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_images(
        cls,
        train_images: Optional[List[Image.Image]] = None,
        train_targets: Optional[Sequence[Any]] = None,
        val_images: Optional[List[Image.Image]] = None,
        val_targets: Optional[Sequence[Any]] = None,
        test_images: Optional[List[Image.Image]] = None,
        test_targets: Optional[Sequence[Any]] = None,
        predict_images: Optional[List[Image.Image]] = None,
        target_formatter: Optional[TargetFormatter] = None,
        input_cls: Type[Input] = ImageClassificationImageInput,
        transform: INPUT_TRANSFORM_TYPE = ImageClassificationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "ImageClassificationData":
        """Load the :class:`~flash.image.classification.data.ImageClassificationData` from lists of PIL images and
        corresponding lists of targets.

        The targets can be in any of our
        :ref:`supported classification target formats <formatting_classification_targets>`.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_images: The list of PIL images to use when training.
            train_targets: The list of targets to use when training.
            val_images: The list of PIL images to use when validating.
            val_targets: The list of targets to use when validating.
            test_images: The list of PIL images to use when testing.
            test_targets: The list of targets to use when testing.
            predict_images: The list of PIL images to use when predicting.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.image.classification.data.ImageClassificationData`.

        Examples
        ________

        .. doctest::

            >>> from PIL import Image
            >>> import numpy as np
            >>> from flash import Trainer
            >>> from flash.image import ImageClassifier, ImageClassificationData
            >>> datamodule = ImageClassificationData.from_images(
            ...     train_images=[
            ...         Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8")),
            ...         Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8")),
            ...         Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8")),
            ...     ],
            ...     train_targets=["cat", "dog", "cat"],
            ...     predict_images=[Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))],
            ...     transform_kwargs=dict(image_size=(128, 128)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['cat', 'dog']
            >>> model = ImageClassifier(backbone="resnet18", num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...
        """
        ds_kw = dict(
            target_formatter=target_formatter,
        )

        train_input = input_cls(RunningStage.TRAINING, train_images, train_targets, **ds_kw)
        ds_kw["target_formatter"] = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_images, val_targets, **ds_kw),
            input_cls(RunningStage.TESTING, test_images, test_targets, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_images, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_tensors(
        cls,
        train_data: Optional[Collection[Tensor]] = None,
        train_targets: Optional[Collection[Any]] = None,
        val_data: Optional[Collection[Tensor]] = None,
        val_targets: Optional[Sequence[Any]] = None,
        test_data: Optional[Collection[Tensor]] = None,
        test_targets: Optional[Sequence[Any]] = None,
        predict_data: Optional[Collection[Tensor]] = None,
        target_formatter: Optional[TargetFormatter] = None,
        input_cls: Type[Input] = ImageClassificationTensorInput,
        transform: INPUT_TRANSFORM_TYPE = ImageClassificationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "ImageClassificationData":
        """Load the :class:`~flash.image.classification.data.ImageClassificationData` from torch tensors (or lists
        of tensors) and corresponding lists of targets.

        The targets can be in any of our
        :ref:`supported classification target formats <formatting_classification_targets>`.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_data: The torch tensor or list of tensors to use when training.
            train_targets: The list of targets to use when training.
            val_data: The torch tensor or list of tensors to use when validating.
            val_targets: The list of targets to use when validating.
            test_data: The torch tensor or list of tensors to use when testing.
            test_targets: The list of targets to use when testing.
            predict_data: The torch tensor or list of tensors to use when predicting.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.image.classification.data.ImageClassificationData`.

        Examples
        ________

        .. doctest::

            >>> import torch
            >>> from flash import Trainer
            >>> from flash.image import ImageClassifier, ImageClassificationData
            >>> datamodule = ImageClassificationData.from_tensors(
            ...     train_data=[torch.rand(3, 64, 64), torch.rand(3, 64, 64), torch.rand(3, 64, 64)],
            ...     train_targets=["cat", "dog", "cat"],
            ...     predict_data=[torch.rand(3, 64, 64)],
            ...     transform_kwargs=dict(image_size=(128, 128)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['cat', 'dog']
            >>> model = ImageClassifier(backbone="resnet18", num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...
        """
        ds_kw = dict(
            target_formatter=target_formatter,
        )

        train_input = input_cls(RunningStage.TRAINING, train_data, train_targets, **ds_kw)
        ds_kw["target_formatter"] = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_data, val_targets, **ds_kw),
            input_cls(RunningStage.TESTING, test_data, test_targets, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_data, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_data_frame(
        cls,
        input_field: str,
        target_fields: Optional[Union[str, Sequence[str]]] = None,
        train_data_frame: Optional[pd.DataFrame] = None,
        train_images_root: Optional[str] = None,
        train_resolver: Optional[Callable[[str, str], str]] = None,
        val_data_frame: Optional[pd.DataFrame] = None,
        val_images_root: Optional[str] = None,
        val_resolver: Optional[Callable[[str, str], str]] = None,
        test_data_frame: Optional[pd.DataFrame] = None,
        test_images_root: Optional[str] = None,
        test_resolver: Optional[Callable[[str, str], str]] = None,
        predict_data_frame: Optional[pd.DataFrame] = None,
        predict_images_root: Optional[str] = None,
        predict_resolver: Optional[Callable[[str, str], str]] = None,
        target_formatter: Optional[TargetFormatter] = None,
        input_cls: Type[Input] = ImageClassificationDataFrameInput,
        transform: INPUT_TRANSFORM_TYPE = ImageClassificationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "ImageClassificationData":
        """Load the :class:`~flash.image.classification.data.ImageClassificationData` from pandas DataFrame objects
        containing image files and their corresponding targets.

        Input images will be extracted from the ``input_field`` in the DataFrame.
        The supported file extensions are: ``.jpg``, ``.jpeg``, ``.png``, ``.ppm``, ``.bmp``, ``.pgm``, ``.tif``,
        ``.tiff``, ``.webp``, and ``.npy``.
        The targets will be extracted from the ``target_fields`` in the DataFrame and can be in any of our
        :ref:`supported classification target formats <formatting_classification_targets>`.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            input_field: The field (column name) in the DataFrames containing the image file paths.
            target_fields: The field (column name) or list of fields in the DataFrames containing the targets.
            train_data_frame: The pandas DataFrame to use when training.
            train_images_root: The root directory containing train images.
            train_resolver: Optionally provide a function which converts an entry from the ``input_field`` into an image
                file path.
            val_data_frame: The pandas DataFrame to use when validating.
            val_images_root: The root directory containing validation images.
            val_resolver: Optionally provide a function which converts an entry from the ``input_field`` into an image
                file path.
            test_data_frame: The pandas DataFrame to use when testing.
            test_images_root: The root directory containing test images.
            test_resolver: Optionally provide a function which converts an entry from the ``input_field`` into an image
                file path.
            predict_data_frame: The pandas DataFrame to use when predicting.
            predict_images_root: The root directory containing predict images.
            predict_resolver: Optionally provide a function which converts an entry from the ``input_field`` into an
                image file path.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.image.classification.data.ImageClassificationData`.

        Examples
        ________

        .. testsetup::

            >>> import os
            >>> from PIL import Image
            >>> rand_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))
            >>> os.makedirs("train_folder", exist_ok=True)
            >>> os.makedirs("predict_folder", exist_ok=True)
            >>> _ = [rand_image.save(os.path.join("train_folder", f"image_{i}.png")) for i in range(1, 4)]
            >>> _ = [rand_image.save(os.path.join("predict_folder", f"predict_image_{i}.png")) for i in range(1, 4)]

        .. doctest::

            >>> from pandas import DataFrame
            >>> from flash import Trainer
            >>> from flash.image import ImageClassifier, ImageClassificationData
            >>> train_data_frame = DataFrame.from_dict(
            ...     {
            ...         "images": ["image_1.png", "image_2.png", "image_3.png"],
            ...         "targets": ["cat", "dog", "cat"],
            ...     }
            ... )
            >>> predict_data_frame = DataFrame.from_dict(
            ...     {
            ...         "images": ["predict_image_1.png", "predict_image_2.png", "predict_image_3.png"],
            ...     }
            ... )
            >>> datamodule = ImageClassificationData.from_data_frame(
            ...     "images",
            ...     "targets",
            ...     train_data_frame=train_data_frame,
            ...     train_images_root="train_folder",
            ...     predict_data_frame=predict_data_frame,
            ...     predict_images_root="predict_folder",
            ...     transform_kwargs=dict(image_size=(128, 128)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['cat', 'dog']
            >>> model = ImageClassifier(backbone="resnet18", num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> import shutil
            >>> shutil.rmtree("train_folder")
            >>> shutil.rmtree("predict_folder")
            >>> del train_data_frame
            >>> del predict_data_frame
        """
        ds_kw = dict(
            target_formatter=target_formatter,
        )

        train_data = (train_data_frame, input_field, target_fields, train_images_root, train_resolver)
        val_data = (val_data_frame, input_field, target_fields, val_images_root, val_resolver)
        test_data = (test_data_frame, input_field, target_fields, test_images_root, test_resolver)
        predict_data = (predict_data_frame, input_field, None, predict_images_root, predict_resolver)

        train_input = input_cls(RunningStage.TRAINING, *train_data, **ds_kw)
        ds_kw["target_formatter"] = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, *val_data, **ds_kw),
            input_cls(RunningStage.TESTING, *test_data, **ds_kw),
            input_cls(RunningStage.PREDICTING, *predict_data, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_csv(
        cls,
        input_field: str,
        target_fields: Optional[Union[str, List[str]]] = None,
        train_file: Optional[PATH_TYPE] = None,
        train_images_root: Optional[PATH_TYPE] = None,
        train_resolver: Optional[Callable[[PATH_TYPE, Any], PATH_TYPE]] = None,
        val_file: Optional[PATH_TYPE] = None,
        val_images_root: Optional[PATH_TYPE] = None,
        val_resolver: Optional[Callable[[PATH_TYPE, Any], PATH_TYPE]] = None,
        test_file: Optional[str] = None,
        test_images_root: Optional[str] = None,
        test_resolver: Optional[Callable[[PATH_TYPE, Any], PATH_TYPE]] = None,
        predict_file: Optional[str] = None,
        predict_images_root: Optional[str] = None,
        predict_resolver: Optional[Callable[[PATH_TYPE, Any], PATH_TYPE]] = None,
        target_formatter: Optional[TargetFormatter] = None,
        input_cls: Type[Input] = ImageClassificationCSVInput,
        transform: INPUT_TRANSFORM_TYPE = ImageClassificationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "ImageClassificationData":
        """Load the :class:`~flash.image.classification.data.ImageClassificationData` from CSV files containing
        image file paths and their corresponding targets.

        Input images will be extracted from the ``input_field`` column in the CSV files.
        The supported file extensions are: ``.jpg``, ``.jpeg``, ``.png``, ``.ppm``, ``.bmp``, ``.pgm``, ``.tif``,
        ``.tiff``, ``.webp``, and ``.npy``.
        The targets will be extracted from the ``target_fields`` in the CSV files and can be in any of our
        :ref:`supported classification target formats <formatting_classification_targets>`.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            input_field: The field (column name) in the CSV files containing the image file paths.
            target_fields: The field (column name) or list of fields in the CSV files containing the targets.
            train_file: The CSV file to use when training.
            train_images_root: The root directory containing train images.
            train_resolver: Optionally provide a function which converts an entry from the ``input_field`` into an image
                file path.
            val_file: The CSV file to use when validating.
            val_images_root: The root directory containing validation images.
            val_resolver: Optionally provide a function which converts an entry from the ``input_field`` into an image
                file path.
            test_file: The CSV file to use when testing.
            test_images_root: The root directory containing test images.
            test_resolver: Optionally provide a function which converts an entry from the ``input_field`` into an image
                file path.
            predict_file: The CSV file to use when predicting.
            predict_images_root: The root directory containing predict images.
            predict_resolver: Optionally provide a function which converts an entry from the ``input_field`` into an
                image file path.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.image.classification.data.ImageClassificationData`.

        Examples
        ________

        The files can be in Comma Separated Values (CSV) format with either a ``.csv`` or ``.txt`` extension.

        .. testsetup::

            >>> import os
            >>> from PIL import Image
            >>> from pandas import DataFrame
            >>> rand_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))
            >>> os.makedirs("train_folder", exist_ok=True)
            >>> os.makedirs("predict_folder", exist_ok=True)
            >>> _ = [rand_image.save(os.path.join("train_folder", f"image_{i}.png")) for i in range(1, 4)]
            >>> _ = [rand_image.save(os.path.join("predict_folder", f"predict_image_{i}.png")) for i in range(1, 4)]
            >>> DataFrame.from_dict({
            ...     "images": ["image_1.png", "image_2.png", "image_3.png"],
            ...     "targets": ["cat", "dog", "cat"],
            ... }).to_csv("train_data.csv", index=False)
            >>> DataFrame.from_dict({
            ...     "images": ["predict_image_1.png", "predict_image_2.png", "predict_image_3.png"],
            ... }).to_csv("predict_data.csv", index=False)

        The file ``train_data.csv`` contains the following:

        .. code-block::

            images,targets
            image_1.png,cat
            image_2.png,dog
            image_3.png,cat

        The file ``predict_data.csv`` contains the following:

        .. code-block::

            images
            predict_image_1.png
            predict_image_2.png
            predict_image_3.png

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.image import ImageClassifier, ImageClassificationData
            >>> datamodule = ImageClassificationData.from_csv(
            ...     "images",
            ...     "targets",
            ...     train_file="train_data.csv",
            ...     train_images_root="train_folder",
            ...     predict_file="predict_data.csv",
            ...     predict_images_root="predict_folder",
            ...     transform_kwargs=dict(image_size=(128, 128)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['cat', 'dog']
            >>> model = ImageClassifier(backbone="resnet18", num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> import shutil
            >>> shutil.rmtree("train_folder")
            >>> shutil.rmtree("predict_folder")
            >>> os.remove("train_data.csv")
            >>> os.remove("predict_data.csv")

        Alternatively, the files can be in Tab Separated Values (TSV) format with a ``.tsv`` extension.

        .. testsetup::

            >>> import os
            >>> from PIL import Image
            >>> from pandas import DataFrame
            >>> rand_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))
            >>> os.makedirs("train_folder", exist_ok=True)
            >>> os.makedirs("predict_folder", exist_ok=True)
            >>> _ = [rand_image.save(os.path.join("train_folder", f"image_{i}.png")) for i in range(1, 4)]
            >>> _ = [rand_image.save(os.path.join("predict_folder", f"predict_image_{i}.png")) for i in range(1, 4)]
            >>> DataFrame.from_dict({
            ...     "images": ["image_1.png", "image_2.png", "image_3.png"],
            ...     "targets": ["cat", "dog", "cat"],
            ... }).to_csv("train_data.tsv", sep="\\t", index=False)
            >>> DataFrame.from_dict({
            ...     "images": ["predict_image_1.png", "predict_image_2.png", "predict_image_3.png"],
            ... }).to_csv("predict_data.tsv", sep="\\t", index=False)

        The file ``train_data.tsv`` contains the following:

        .. code-block::

            images      targets
            image_1.png cat
            image_2.png dog
            image_3.png cat

        The file ``predict_data.tsv`` contains the following:

        .. code-block::

            images
            predict_image_1.png
            predict_image_2.png
            predict_image_3.png

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.image import ImageClassifier, ImageClassificationData
            >>> datamodule = ImageClassificationData.from_csv(
            ...     "images",
            ...     "targets",
            ...     train_file="train_data.tsv",
            ...     train_images_root="train_folder",
            ...     predict_file="predict_data.tsv",
            ...     predict_images_root="predict_folder",
            ...     transform_kwargs=dict(image_size=(128, 128)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['cat', 'dog']
            >>> model = ImageClassifier(backbone="resnet18", num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> import shutil
            >>> shutil.rmtree("train_folder")
            >>> shutil.rmtree("predict_folder")
            >>> os.remove("train_data.tsv")
            >>> os.remove("predict_data.tsv")
        """
        ds_kw = dict(
            target_formatter=target_formatter,
        )

        train_data = (train_file, input_field, target_fields, train_images_root, train_resolver)
        val_data = (val_file, input_field, target_fields, val_images_root, val_resolver)
        test_data = (test_file, input_field, target_fields, test_images_root, test_resolver)
        predict_data = (predict_file, input_field, None, predict_images_root, predict_resolver)

        train_input = input_cls(RunningStage.TRAINING, *train_data, **ds_kw)
        ds_kw["target_formatter"] = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, *val_data, **ds_kw),
            input_cls(RunningStage.TESTING, *test_data, **ds_kw),
            input_cls(RunningStage.PREDICTING, *predict_data, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    @requires("fiftyone")
    def from_fiftyone(
        cls,
        train_dataset: Optional[SampleCollection] = None,
        val_dataset: Optional[SampleCollection] = None,
        test_dataset: Optional[SampleCollection] = None,
        predict_dataset: Optional[SampleCollection] = None,
        label_field: str = "ground_truth",
        target_formatter: Optional[TargetFormatter] = None,
        input_cls: Type[Input] = ImageClassificationFiftyOneInput,
        transform: INPUT_TRANSFORM_TYPE = ImageClassificationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs,
    ) -> "ImageClassificationData":
        """Load the :class:`~flash.image.classification.data.ImageClassificationData` from FiftyOne
        ``SampleCollection`` objects.

        The supported file extensions are: ``.jpg``, ``.jpeg``, ``.png``, ``.ppm``, ``.bmp``, ``.pgm``, ``.tif``,
        ``.tiff``, ``.webp``, and ``.npy``.
        The targets will be extracted from the ``label_field`` in the ``SampleCollection`` objects and can be in any
        of our :ref:`supported classification target formats <formatting_classification_targets>`.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_dataset: The ``SampleCollection`` to use when training.
            val_dataset: The ``SampleCollection`` to use when validating.
            test_dataset: The ``SampleCollection`` to use when testing.
            predict_dataset: The ``SampleCollection`` to use when predicting.
            label_field: The field in the ``SampleCollection`` objects containing the targets.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.image.classification.data.ImageClassificationData`.

        Examples
        ________

        .. testsetup::

            >>> from PIL import Image
            >>> rand_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))
            >>> _ = [rand_image.save(f"image_{i}.png") for i in range(1, 4)]
            >>> _ = [rand_image.save(f"predict_image_{i}.png") for i in range(1, 4)]

        .. doctest::

            >>> import fiftyone as fo
            >>> from flash import Trainer
            >>> from flash.image import ImageClassifier, ImageClassificationData
            >>> train_dataset = fo.Dataset.from_images(
            ...     ["image_1.png", "image_2.png", "image_3.png"]
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            <BLANKLINE>
            ...
            >>> samples = [train_dataset[filepath] for filepath in train_dataset.values("filepath")]
            >>> for sample, label in zip(samples, ["cat", "dog", "cat"]):
            ...     sample["ground_truth"] = fo.Classification(label=label)
            ...     sample.save()
            ...
            >>> predict_dataset = fo.Dataset.from_images(
            ...     ["predict_image_1.png", "predict_image_2.png", "predict_image_3.png"]
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            <BLANKLINE>
            ...
            >>> datamodule = ImageClassificationData.from_fiftyone(
            ...     train_dataset=train_dataset,
            ...     predict_dataset=predict_dataset,
            ...     transform_kwargs=dict(image_size=(128, 128)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['cat', 'dog']
            >>> model = ImageClassifier(backbone="resnet18", num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> import os
            >>> _ = [os.remove(f"image_{i}.png") for i in range(1, 4)]
            >>> _ = [os.remove(f"predict_image_{i}.png") for i in range(1, 4)]
            >>> del train_dataset
            >>> del predict_dataset
        """
        ds_kw = dict(
            target_formatter=target_formatter,
        )

        train_input = input_cls(RunningStage.TRAINING, train_dataset, label_field=label_field, **ds_kw)
        ds_kw["target_formatter"] = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_dataset, label_field=label_field, **ds_kw),
            input_cls(RunningStage.TESTING, test_dataset, label_field=label_field, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_dataset, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_labelstudio(
        cls,
        export_json: str = None,
        train_export_json: str = None,
        val_export_json: str = None,
        test_export_json: str = None,
        predict_export_json: str = None,
        data_folder: str = None,
        train_data_folder: str = None,
        val_data_folder: str = None,
        test_data_folder: str = None,
        predict_data_folder: str = None,
        input_cls: Type[Input] = LabelStudioImageClassificationInput,
        transform: INPUT_TRANSFORM_TYPE = ImageClassificationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        val_split: Optional[float] = None,
        multi_label: Optional[bool] = False,
        **data_module_kwargs: Any,
    ) -> "ImageClassificationData":
        """Creates a :class:`~flash.core.data.data_module.DataModule` object
        from the given export file and data directory using the
        :class:`~flash.core.data.io.input.Input` of name
        :attr:`~flash.core.data.io.input.InputFormat.FOLDERS`
        from the passed or constructed :class:`~flash.core.data.io.input_transform.InputTransform`.

        Args:
            export_json: path to label studio export file
            train_export_json: path to label studio export file for train set.(overrides export_json if specified)
            val_export_json: path to label studio export file for validation
            test_export_json: path to label studio export file for test
            predict_export_json: path to label studio export file for predict
            data_folder: path to label studio data folder
            train_data_folder: path to label studio data folder for train data set.(overrides data_folder if specified)
            val_data_folder: path to label studio data folder for validation data
            test_data_folder: path to label studio data folder for test data
            predict_data_folder: path to label studio data folder for predict data
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            multi_label: Whether the labels are multi encoded
            data_module_kwargs: Additional keyword arguments to use when constructing the datamodule.

        Returns:
            The constructed data module.

        Examples::

            data_module = DataModule.from_labelstudio(
                export_json='project.json',
                data_folder='label-studio/media/upload',
                val_split=0.8,
            )
        """

        train_data, val_data, test_data, predict_data = _parse_labelstudio_arguments(
            export_json=export_json,
            train_export_json=train_export_json,
            val_export_json=val_export_json,
            test_export_json=test_export_json,
            predict_export_json=predict_export_json,
            data_folder=data_folder,
            train_data_folder=train_data_folder,
            val_data_folder=val_data_folder,
            test_data_folder=test_data_folder,
            predict_data_folder=predict_data_folder,
            val_split=val_split,
            multi_label=multi_label,
        )

        ds_kw = dict()

        train_input = input_cls(RunningStage.TRAINING, train_data, **ds_kw)
        ds_kw["parameters"] = getattr(train_input, "parameters", None)

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_data, **ds_kw),
            input_cls(RunningStage.TESTING, val_data, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_data, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_datasets(
        cls,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        predict_dataset: Optional[Dataset] = None,
        input_cls: Type[Input] = DatasetInput,
        transform: INPUT_TRANSFORM_TYPE = ImageClassificationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "DataModule":
        """Creates a :class:`~flash.core.data.data_module.DataModule` object from the given datasets using the
        :class:`~flash.core.data.io.input.Input`
        of name :attr:`~flash.core.data.io.input.InputFormat.DATASETS`
        from the passed or constructed :class:`~flash.core.data.io.input_transform.InputTransform`.

        Args:
            train_dataset: Dataset used during training.
            val_dataset: Dataset used during validating.
            test_dataset: Dataset used during testing.
            predict_dataset: Dataset used during predicting.
            input_cls: Input class used to create the datasets.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Additional keyword arguments to be used when constructing the transform.
            data_module_kwargs: Additional keyword arguments to use when constructing the DataModule.

        Returns:
            The constructed data module.

        Examples::

            data_module = DataModule.from_datasets(
                train_dataset=train_dataset,
            )
        """
        ds_kw = dict()

        return cls(
            input_cls(RunningStage.TRAINING, train_dataset, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_dataset, **ds_kw),
            input_cls(RunningStage.TESTING, test_dataset, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_dataset, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    def set_block_viz_window(self, value: bool) -> None:
        """Setter method to switch on/off matplotlib to pop up windows."""
        self.data_fetcher.block_viz_window = value

    @staticmethod
    def configure_data_fetcher(*args, **kwargs) -> BaseDataFetcher:
        return MatplotlibVisualization(*args, **kwargs)


class MatplotlibVisualization(BaseVisualization):
    """Process and show the image batch and its associated label using matplotlib."""

    max_cols: int = 4  # maximum number of columns we accept
    block_viz_window: bool = True  # parameter to allow user to block visualisation windows

    @staticmethod
    @requires("image")
    def _to_numpy(img: Union[np.ndarray, Tensor, Image.Image]) -> np.ndarray:
        out: np.ndarray
        if isinstance(img, np.ndarray):
            out = img
        elif isinstance(img, Image.Image):
            out = np.array(img)
        elif isinstance(img, Tensor):
            out = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        else:
            raise TypeError(f"Unknown image type. Got: {type(img)}.")
        return out

    @requires("matplotlib")
    def _show_images_and_labels(
        self,
        data: List[Any],
        num_samples: int,
        title: str,
        limit_nb_samples: int = None,
        figsize: Tuple[int, int] = (6.4, 4.8),
    ):
        num_samples = max(1, min(num_samples, limit_nb_samples))

        # define the image grid
        cols: int = min(num_samples, self.max_cols)
        rows: int = num_samples // cols

        # create figure and set title
        fig, axs = plt.subplots(rows, cols, figsize=figsize)
        fig.suptitle(title)

        if not isinstance(axs, np.ndarray):
            axs = np.array(axs)
        axs = axs.flatten()

        for i, ax in enumerate(axs):
            # unpack images and labels
            if isinstance(data, list):
                _img, _label = data[i][DataKeys.INPUT], data[i].get(DataKeys.TARGET, "")
            elif isinstance(data, dict):
                _img, _label = data[DataKeys.INPUT][i], data.get(DataKeys.TARGET, [""] * (i + 1))[i]
            else:
                raise TypeError(f"Unknown data type. Got: {type(data)}.")
            # convert images to numpy
            _img: np.ndarray = self._to_numpy(_img)
            if isinstance(_label, Tensor):
                _label = _label.squeeze().tolist()
            # show image and set label as subplot title
            ax.imshow(_img)
            ax.set_title(str(_label))
            ax.axis("off")
        plt.show(block=self.block_viz_window)

    def show_load_sample(
        self,
        samples: List[Any],
        running_stage: RunningStage,
        limit_nb_samples: int = None,
        figsize: Tuple[int, int] = (6.4, 4.8),
    ):
        win_title: str = f"{running_stage} - show_load_sample"
        self._show_images_and_labels(samples, len(samples), win_title, limit_nb_samples, figsize)

    def show_per_sample_transform(
        self,
        samples: List[Any],
        running_stage: RunningStage,
        limit_nb_samples: int = None,
        figsize: Tuple[int, int] = (6.4, 4.8),
    ):
        win_title: str = f"{running_stage} - show_per_sample_transform"
        self._show_images_and_labels(samples, len(samples), win_title, limit_nb_samples, figsize)

    def show_per_batch_transform(
        self, batch: List[Any], running_stage, limit_nb_samples: int = None, figsize: Tuple[int, int] = (6.4, 4.8)
    ):
        win_title: str = f"{running_stage} - show_per_batch_transform"
        self._show_images_and_labels(batch[0], batch[0][DataKeys.INPUT].shape[0], win_title, limit_nb_samples, figsize)
