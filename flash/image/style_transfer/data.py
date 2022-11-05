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
from typing import Any, Collection, Dict, Optional, Sequence, Type

import numpy as np
from torch import Tensor

from flash.core.data.data_module import DataModule
from flash.core.data.io.input import Input
from flash.core.utilities.imports import _IMAGE_TESTING
from flash.core.utilities.stability import beta
from flash.core.utilities.stages import RunningStage
from flash.core.utilities.types import INPUT_TRANSFORM_TYPE
from flash.image.classification.input import ImageClassificationFilesInput, ImageClassificationFolderInput
from flash.image.data import ImageNumpyInput, ImageTensorInput
from flash.image.style_transfer.input_transform import StyleTransferInputTransform

# Skip doctests if requirements aren't available
if not _IMAGE_TESTING:
    __doctest_skip__ = ["StyleTransferData", "StyleTransferData.*"]


@beta("Style transfer is currently in Beta.")
class StyleTransferData(DataModule):
    """The ``StyleTransferData`` class is a :class:`~flash.core.data.data_module.DataModule` with a set of
    classmethods for loading data for image style transfer."""

    input_transform_cls = StyleTransferInputTransform

    @classmethod
    def from_files(
        cls,
        train_files: Optional[Sequence[str]] = None,
        predict_files: Optional[Sequence[str]] = None,
        input_cls: Type[Input] = ImageClassificationFilesInput,
        transform: INPUT_TRANSFORM_TYPE = StyleTransferInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any
    ) -> "StyleTransferData":
        """Load the :class:`~flash.image.style_transfer.data.StyleTransferData` from lists of image files.

        The supported file extensions are: ``.jpg``, ``.jpeg``, ``.png``, ``.ppm``, ``.bmp``, ``.pgm``, ``.tif``,
        ``.tiff``, ``.webp``, and ``.npy``.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_files: The list of image files to use when training.
            predict_files: The list of image files to use when predicting.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
              :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.image.style_transfer.data.StyleTransferData`.

        Examples
        ________

        .. testsetup::

            >>> from PIL import Image
            >>> rand_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))
            >>> _ = [rand_image.save(f"image_{i}.png") for i in range(1, 4)]
            >>> _ = [rand_image.save(f"predict_image_{i}.png") for i in range(1, 4)]

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.image import StyleTransfer, StyleTransferData
            >>> datamodule = StyleTransferData.from_files(
            ...     train_files=["image_1.png", "image_2.png", "image_3.png"],
            ...     predict_files=["predict_image_1.png", "predict_image_2.png", "predict_image_3.png"],
            ...     transform_kwargs=dict(image_size=(128, 128)),
            ...     batch_size=2,
            ... )
            >>> model = StyleTransfer()
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
            input_cls(RunningStage.TRAINING, train_files),
            predict_input=input_cls(RunningStage.PREDICTING, predict_files),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[str] = None,
        predict_folder: Optional[str] = None,
        input_cls: Type[Input] = ImageClassificationFolderInput,
        transform: INPUT_TRANSFORM_TYPE = StyleTransferInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any
    ) -> "StyleTransferData":
        """Load the :class:`~flash.image.style_transfer.data.StyleTransferData` from folders containing images.

        The supported file extensions are: ``.jpg``, ``.jpeg``, ``.png``, ``.ppm``, ``.bmp``, ``.pgm``, ``.tif``,
        ``.tiff``, ``.webp``, and ``.npy``.
        Here's the required folder structure:

        .. code-block::

            train_folder
            ├── image_1.png
            ├── image_2.png
            ├── image_3.png
            ...

        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_folder: The folder containing images to use when training.
            predict_folder: The folder containing images to use when predicting.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
              :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.image.style_transfer.data.StyleTransferData`.

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

            >>> from flash import Trainer
            >>> from flash.image import StyleTransfer, StyleTransferData
            >>> datamodule = StyleTransferData.from_folders(
            ...     train_folder="train_folder",
            ...     predict_folder="predict_folder",
            ...     transform_kwargs=dict(image_size=(128, 128)),
            ...     batch_size=2,
            ... )
            >>> model = StyleTransfer()
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

        return cls(
            input_cls(RunningStage.TRAINING, train_folder),
            predict_input=input_cls(RunningStage.PREDICTING, predict_folder),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_numpy(
        cls,
        train_data: Optional[Collection[np.ndarray]] = None,
        predict_data: Optional[Collection[np.ndarray]] = None,
        input_cls: Type[Input] = ImageNumpyInput,
        transform: INPUT_TRANSFORM_TYPE = StyleTransferInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any
    ) -> "StyleTransferData":
        """Load the :class:`~flash.image.style_transfer.data.StyleTransferData` from numpy arrays (or lists of
        arrays).

        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_data: The numpy array or list of arrays to use when training.
            predict_data: The numpy array or list of arrays to use when predicting.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
              :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.image.style_transfer.data.StyleTransferData`.

        Examples
        ________

        .. doctest::

            >>> import numpy as np
            >>> from flash import Trainer
            >>> from flash.image import StyleTransfer, StyleTransferData
            >>> datamodule = StyleTransferData.from_numpy(
            ...     train_data=[np.random.rand(3, 64, 64), np.random.rand(3, 64, 64), np.random.rand(3, 64, 64)],
            ...     predict_data=[np.random.rand(3, 64, 64)],
            ...     transform_kwargs=dict(image_size=(128, 128)),
            ...     batch_size=2,
            ... )
            >>> model = StyleTransfer()
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...
        """

        return cls(
            input_cls(RunningStage.TRAINING, train_data),
            predict_input=input_cls(RunningStage.PREDICTING, predict_data),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_tensors(
        cls,
        train_data: Optional[Collection[Tensor]] = None,
        predict_data: Optional[Collection[Tensor]] = None,
        input_cls: Type[Input] = ImageTensorInput,
        transform: INPUT_TRANSFORM_TYPE = StyleTransferInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any
    ) -> "StyleTransferData":
        """Load the :class:`~flash.image.style_transfer.data.StyleTransferData` from torch tensors (or lists of
        tensors).

        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_data: The torch tensor or list of tensors to use when training.
            predict_data: The torch tensor or list of tensors to use when predicting.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
              :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.image.style_transfer.data.StyleTransferData`.

        Examples
        ________

        .. doctest::

            >>> import torch
            >>> from flash import Trainer
            >>> from flash.image import StyleTransfer, StyleTransferData
            >>> datamodule = StyleTransferData.from_tensors(
            ...     train_data=[torch.rand(3, 64, 64), torch.rand(3, 64, 64), torch.rand(3, 64, 64)],
            ...     predict_data=[torch.rand(3, 64, 64)],
            ...     transform_kwargs=dict(image_size=(128, 128)),
            ...     batch_size=2,
            ... )
            >>> model = StyleTransfer()
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...
        """

        return cls(
            input_cls(RunningStage.TRAINING, train_data),
            predict_input=input_cls(RunningStage.PREDICTING, predict_data),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )
