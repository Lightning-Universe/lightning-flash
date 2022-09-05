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
from functools import partial
from typing import Any, Callable, Collection, Dict, List, Optional, Sequence, Type, Union

import numpy as np
import torch

from flash.core.data.data_module import DataModule
from flash.core.data.io.input import Input
from flash.core.data.utilities.classification import TargetFormatter
from flash.core.data.utilities.sort import sorted_alphanumeric
from flash.core.integrations.icevision.data import IceVisionInput
from flash.core.integrations.icevision.transforms import IceVisionInputTransform
from flash.core.utilities.imports import (
    _FIFTYONE_AVAILABLE,
    _ICEVISION_AVAILABLE,
    _IMAGE_EXTRAS_TESTING,
    Image,
    requires,
)
from flash.core.utilities.stages import RunningStage
from flash.core.utilities.types import INPUT_TRANSFORM_TYPE
from flash.image.detection.input import (
    ObjectDetectionFiftyOneInput,
    ObjectDetectionFilesInput,
    ObjectDetectionImageInput,
    ObjectDetectionNumpyInput,
    ObjectDetectionTensorInput,
)

if _FIFTYONE_AVAILABLE:
    SampleCollection = "fiftyone.core.collections.SampleCollection"
else:
    SampleCollection = None

if _ICEVISION_AVAILABLE:
    from icevision.core import ClassMap
    from icevision.parsers import COCOBBoxParser, Parser, VIABBoxParser, VOCBBoxParser
else:
    COCOBBoxParser = object
    VIABBoxParser = object
    VOCBBoxParser = object
    Parser = object

# Skip doctests if requirements aren't available
if not _IMAGE_EXTRAS_TESTING:
    __doctest_skip__ = ["ObjectDetectionData", "ObjectDetectionData.*"]


class ObjectDetectionData(DataModule):
    """The ``ObjectDetectionData`` class is a :class:`~flash.core.data.data_module.DataModule` with a set of
    classmethods for loading data for object detection."""

    input_transform_cls = IceVisionInputTransform

    @classmethod
    def from_files(
        cls,
        train_files: Optional[Sequence[str]] = None,
        train_targets: Optional[Sequence[Sequence[Any]]] = None,
        train_bboxes: Optional[Sequence[Sequence[Dict[str, int]]]] = None,
        val_files: Optional[Sequence[str]] = None,
        val_targets: Optional[Sequence[Sequence[Any]]] = None,
        val_bboxes: Optional[Sequence[Sequence[Dict[str, int]]]] = None,
        test_files: Optional[Sequence[str]] = None,
        test_targets: Optional[Sequence[Sequence[Any]]] = None,
        test_bboxes: Optional[Sequence[Sequence[Dict[str, int]]]] = None,
        predict_files: Optional[Sequence[str]] = None,
        target_formatter: Optional[TargetFormatter] = None,
        input_cls: Type[Input] = ObjectDetectionFilesInput,
        transform: INPUT_TRANSFORM_TYPE = IceVisionInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "ObjectDetectionData":
        """Creates a :class:`~flash.image.detection.data.ObjectDetectionData` object from the given data list of
        image files, bounding boxes, and targets.

        The supported file extensions are: ``.jpg``, ``.jpeg``, ``.png``, ``.ppm``, ``.bmp``, ``.pgm``, ``.tif``,
        ``.tiff``, ``.webp``, and ``.npy``.
        The targets can be in any of our
        :ref:`supported classification target formats <formatting_classification_targets>`.
        The bounding boxes are expected to be dictionaries with integer values (representing pixels) and the following
        keys: ``xmin``, ``ymin``, ``width``, ``height``.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_files: The list of image files to use when training.
            train_targets: The list of lists of targets to use when training.
            train_bboxes: The list of lists of bounding boxes to use when training.
            val_files: The list of image files to use when validating.
            val_targets: The list of lists of targets to use when validating.
            val_bboxes: The list of lists of bounding boxes to use when validating.
            test_files: The list of image files to use when testing.
            test_targets: The list of lists of targets to use when testing.
            test_bboxes: The list of lists of bounding boxes to use when testing.
            predict_files: The list of image files to use when predicting.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.image.detection.data.ObjectDetectionData`.

        Examples
        ________

        .. testsetup::

            >>> import numpy as np
            >>> from PIL import Image
            >>> rand_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))
            >>> _ = [rand_image.save(f"image_{i}.png") for i in range(1, 4)]
            >>> _ = [rand_image.save(f"predict_image_{i}.png") for i in range(1, 4)]

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.image import ObjectDetector, ObjectDetectionData
            >>> datamodule = ObjectDetectionData.from_files(
            ...     train_files=["image_1.png", "image_2.png", "image_3.png"],
            ...     train_targets=[["cat"], ["dog"], ["cat"]],
            ...     train_bboxes=[
            ...         [{"xmin": 10, "ymin": 20, "width": 5, "height": 10}],
            ...         [{"xmin": 20, "ymin": 30, "width": 10, "height": 10}],
            ...         [{"xmin": 10, "ymin": 20, "width": 5, "height": 25}],
            ...     ],
            ...     predict_files=["predict_image_1.png", "predict_image_2.png", "predict_image_3.png"],
            ...     transform_kwargs=dict(image_size=(128, 128)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            3
            >>> datamodule.labels
            ['background', 'cat', 'dog']
            >>> model = ObjectDetector(labels=datamodule.labels)
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

        train_input = input_cls(
            RunningStage.TRAINING,
            train_files,
            train_targets,
            train_bboxes,
            **ds_kw,
        )
        ds_kw["target_formatter"] = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(
                RunningStage.VALIDATING,
                val_files,
                val_targets,
                val_bboxes,
                **ds_kw,
            ),
            input_cls(
                RunningStage.TESTING,
                test_files,
                test_targets,
                test_bboxes,
                **ds_kw,
            ),
            input_cls(RunningStage.PREDICTING, predict_files, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_numpy(
        cls,
        train_data: Optional[Collection[np.ndarray]] = None,
        train_targets: Optional[Sequence[Sequence[Any]]] = None,
        train_bboxes: Optional[Sequence[Sequence[Dict[str, int]]]] = None,
        val_data: Optional[Collection[np.ndarray]] = None,
        val_targets: Optional[Sequence[Sequence[Any]]] = None,
        val_bboxes: Optional[Sequence[Sequence[Dict[str, int]]]] = None,
        test_data: Optional[Collection[np.ndarray]] = None,
        test_targets: Optional[Sequence[Sequence[Any]]] = None,
        test_bboxes: Optional[Sequence[Sequence[Dict[str, int]]]] = None,
        predict_data: Optional[Collection[np.ndarray]] = None,
        target_formatter: Optional[TargetFormatter] = None,
        input_cls: Type[Input] = ObjectDetectionNumpyInput,
        transform: INPUT_TRANSFORM_TYPE = IceVisionInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "ObjectDetectionData":
        """Creates a :class:`~flash.image.detection.data.ObjectDetectionData` object from the given from numpy
        arrays (or lists of arrays) and corresponding lists of bounding boxes and targets.

        The targets can be in any of our
        :ref:`supported classification target formats <formatting_classification_targets>`.
        The bounding boxes are expected to be dictionaries with integer values (representing pixels) and the following
        keys: ``xmin``, ``ymin``, ``width``, ``height``.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_data: The numpy array or list of arrays to use when training.
            train_targets: The list of lists of targets to use when training.
            train_bboxes: The list of lists of bounding boxes to use when training.
            val_data: The numpy array or list of arrays to use when validating.
            val_targets: The list of lists of targets to use when validating.
            val_bboxes: The list of lists of bounding boxes to use when validating.
            test_data: The numpy array or list of arrays to use when testing.
            test_targets: The list of lists of targets to use when testing.
            test_bboxes: The list of lists of bounding boxes to use when testing.
            predict_data: The numpy array or list of arrays to use when predicting.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.image.detection.data.ObjectDetectionData`.

        Examples
        ________

        .. doctest::

            >>> import numpy as np
            >>> from flash import Trainer
            >>> from flash.image import ObjectDetector, ObjectDetectionData
            >>> datamodule = ObjectDetectionData.from_numpy(
            ...     train_data=[np.random.rand(3, 64, 64), np.random.rand(3, 64, 64), np.random.rand(3, 64, 64)],
            ...     train_targets=[["cat"], ["dog"], ["cat"]],
            ...     train_bboxes=[
            ...         [{"xmin": 10, "ymin": 20, "width": 5, "height": 10}],
            ...         [{"xmin": 20, "ymin": 30, "width": 10, "height": 10}],
            ...         [{"xmin": 10, "ymin": 20, "width": 5, "height": 25}],
            ...     ],
            ...     predict_data=[np.random.rand(3, 64, 64)],
            ...     transform_kwargs=dict(image_size=(128, 128)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            3
            >>> datamodule.labels
            ['background', 'cat', 'dog']
            >>> model = ObjectDetector(labels=datamodule.labels)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...
        """

        ds_kw = dict(
            target_formatter=target_formatter,
        )

        train_input = input_cls(
            RunningStage.TRAINING,
            train_data,
            train_targets,
            train_bboxes,
            **ds_kw,
        )
        ds_kw["target_formatter"] = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(
                RunningStage.VALIDATING,
                val_data,
                val_targets,
                val_bboxes,
                **ds_kw,
            ),
            input_cls(
                RunningStage.TESTING,
                test_data,
                test_targets,
                test_bboxes,
                **ds_kw,
            ),
            input_cls(RunningStage.PREDICTING, predict_data, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_images(
        cls,
        train_images: Optional[List[Image.Image]] = None,
        train_targets: Optional[Sequence[Sequence[Any]]] = None,
        train_bboxes: Optional[Sequence[Sequence[Dict[str, int]]]] = None,
        val_images: Optional[List[Image.Image]] = None,
        val_targets: Optional[Sequence[Sequence[Any]]] = None,
        val_bboxes: Optional[Sequence[Sequence[Dict[str, int]]]] = None,
        test_images: Optional[List[Image.Image]] = None,
        test_targets: Optional[Sequence[Sequence[Any]]] = None,
        test_bboxes: Optional[Sequence[Sequence[Dict[str, int]]]] = None,
        predict_images: Optional[List[Image.Image]] = None,
        target_formatter: Optional[TargetFormatter] = None,
        input_cls: Type[Input] = ObjectDetectionImageInput,
        transform: INPUT_TRANSFORM_TYPE = IceVisionInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "ObjectDetectionData":
        """Creates a :class:`~flash.image.detection.data.ObjectDetectionData` object from the given lists of PIL
        images and corresponding lists of bounding boxes and targets.

        The targets can be in any of our
        :ref:`supported classification target formats <formatting_classification_targets>`.
        The bounding boxes are expected to be dictionaries with integer values (representing pixels) and the following
        keys: ``xmin``, ``ymin``, ``width``, ``height``.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_images: The list of PIL images to use when training.
            train_targets: The list of lists of targets to use when training.
            train_bboxes: The list of lists of bounding boxes to use when training.
            val_images: The list of PIL images to use when validating.
            val_targets: The list of lists of targets to use when validating.
            val_bboxes: The list of lists of bounding boxes to use when validating.
            test_images: The list of PIL images to use when testing.
            test_targets: The list of lists of targets to use when testing.
            test_bboxes: The list of lists of bounding boxes to use when testing.
            predict_images: The list of PIL images to use when predicting.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.image.detection.data.ObjectDetectionData`.

        Examples
        ________

        .. doctest::

            >>> from PIL import Image
            >>> import numpy as np
            >>> from flash import Trainer
            >>> from flash.image import ObjectDetector, ObjectDetectionData
            >>> datamodule = ObjectDetectionData.from_images(
            ...     train_images=[
            ...         Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8")),
            ...         Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8")),
            ...         Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8")),
            ...     ],
            ...     train_targets=[["cat"], ["dog"], ["cat"]],
            ...     train_bboxes=[
            ...         [{"xmin": 10, "ymin": 20, "width": 5, "height": 10}],
            ...         [{"xmin": 20, "ymin": 30, "width": 10, "height": 10}],
            ...         [{"xmin": 10, "ymin": 20, "width": 5, "height": 25}],
            ...     ],
            ...     predict_images=[Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))],
            ...     transform_kwargs=dict(image_size=(128, 128)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            3
            >>> datamodule.labels
            ['background', 'cat', 'dog']
            >>> model = ObjectDetector(labels=datamodule.labels)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...
        """

        ds_kw = dict(
            target_formatter=target_formatter,
        )

        train_input = input_cls(
            RunningStage.TRAINING,
            train_images,
            train_targets,
            train_bboxes,
            **ds_kw,
        )
        ds_kw["target_formatter"] = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(
                RunningStage.VALIDATING,
                val_images,
                val_targets,
                val_bboxes,
                **ds_kw,
            ),
            input_cls(
                RunningStage.TESTING,
                test_images,
                test_targets,
                test_bboxes,
                **ds_kw,
            ),
            input_cls(RunningStage.PREDICTING, predict_images, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_tensors(
        cls,
        train_data: Optional[Collection[torch.Tensor]] = None,
        train_targets: Optional[Sequence[Sequence[Any]]] = None,
        train_bboxes: Optional[Sequence[Sequence[Dict[str, int]]]] = None,
        val_data: Optional[Collection[torch.Tensor]] = None,
        val_targets: Optional[Sequence[Sequence[Any]]] = None,
        val_bboxes: Optional[Sequence[Sequence[Dict[str, int]]]] = None,
        test_data: Optional[Collection[torch.Tensor]] = None,
        test_targets: Optional[Sequence[Sequence[Any]]] = None,
        test_bboxes: Optional[Sequence[Sequence[Dict[str, int]]]] = None,
        predict_data: Optional[Collection[torch.Tensor]] = None,
        target_formatter: Optional[TargetFormatter] = None,
        input_cls: Type[Input] = ObjectDetectionTensorInput,
        transform: INPUT_TRANSFORM_TYPE = IceVisionInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "ObjectDetectionData":
        """Creates a :class:`~flash.image.detection.data.ObjectDetectionData` object from the given from torch
        tensors (or lists of tensors) and corresponding lists of bounding boxes and targets.

        The targets can be in any of our
        :ref:`supported classification target formats <formatting_classification_targets>`.
        The bounding boxes are expected to be dictionaries with integer values (representing pixels) and the following
        keys: ``xmin``, ``ymin``, ``width``, ``height``.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_data: The torch tensor or list of tensors to use when training.
            train_targets: The list of lists of targets to use when training.
            train_bboxes: The list of lists of bounding boxes to use when training.
            val_data: The torch tensor or list of tensors to use when validating.
            val_targets: The list of lists of targets to use when validating.
            val_bboxes: The list of lists of bounding boxes to use when validating.
            test_data: The torch tensor or list of tensors to use when testing.
            test_targets: The list of lists of targets to use when testing.
            test_bboxes: The list of lists of bounding boxes to use when testing.
            predict_data: The torch tensor or list of tensors to use when predicting.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.image.detection.data.ObjectDetectionData`.

        Examples
        ________

        .. doctest::

            >>> import torch
            >>> from flash import Trainer
            >>> from flash.image import ObjectDetector, ObjectDetectionData
            >>> datamodule = ObjectDetectionData.from_tensors(
            ...     train_data=[torch.rand(3, 64, 64), torch.rand(3, 64, 64), torch.rand(3, 64, 64)],
            ...     train_targets=[["cat"], ["dog"], ["cat"]],
            ...     train_bboxes=[
            ...         [{"xmin": 10, "ymin": 20, "width": 5, "height": 10}],
            ...         [{"xmin": 20, "ymin": 30, "width": 10, "height": 10}],
            ...         [{"xmin": 10, "ymin": 20, "width": 5, "height": 25}],
            ...     ],
            ...     predict_data=[torch.rand(3, 64, 64)],
            ...     transform_kwargs=dict(image_size=(128, 128)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            3
            >>> datamodule.labels
            ['background', 'cat', 'dog']
            >>> model = ObjectDetector(labels=datamodule.labels)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...
        """

        ds_kw = dict(
            target_formatter=target_formatter,
        )

        train_input = input_cls(
            RunningStage.TRAINING,
            train_data,
            train_targets,
            train_bboxes,
            **ds_kw,
        )
        ds_kw["target_formatter"] = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(
                RunningStage.VALIDATING,
                val_data,
                val_targets,
                val_bboxes,
                **ds_kw,
            ),
            input_cls(
                RunningStage.TESTING,
                test_data,
                test_targets,
                test_bboxes,
                **ds_kw,
            ),
            input_cls(RunningStage.PREDICTING, predict_data, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_icedata(
        cls,
        train_folder: Optional[str] = None,
        train_ann_file: Optional[str] = None,
        train_parser_kwargs: Optional[Dict[str, Any]] = None,
        val_folder: Optional[str] = None,
        val_ann_file: Optional[str] = None,
        val_parser_kwargs: Optional[Dict[str, Any]] = None,
        test_folder: Optional[str] = None,
        test_ann_file: Optional[str] = None,
        test_parser_kwargs: Optional[Dict[str, Any]] = None,
        predict_folder: Optional[str] = None,
        transform: INPUT_TRANSFORM_TYPE = IceVisionInputTransform,
        parser: Optional[Union[Callable, Type[Parser]]] = None,
        input_cls: Type[Input] = IceVisionInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs,
    ) -> "ObjectDetectionData":

        ds_kw = dict(parser=parser)

        return cls(
            input_cls(
                RunningStage.TRAINING,
                train_folder,
                train_ann_file,
                parser_kwargs=train_parser_kwargs,
                **ds_kw,
            ),
            input_cls(
                RunningStage.VALIDATING,
                val_folder,
                val_ann_file,
                parser_kwargs=val_parser_kwargs,
                **ds_kw,
            ),
            input_cls(
                RunningStage.TESTING,
                test_folder,
                test_ann_file,
                parser_kwargs=test_parser_kwargs,
                **ds_kw,
            ),
            input_cls(RunningStage.PREDICTING, predict_folder, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_coco(
        cls,
        train_folder: Optional[str] = None,
        train_ann_file: Optional[str] = None,
        val_folder: Optional[str] = None,
        val_ann_file: Optional[str] = None,
        test_folder: Optional[str] = None,
        test_ann_file: Optional[str] = None,
        predict_folder: Optional[str] = None,
        transform: INPUT_TRANSFORM_TYPE = IceVisionInputTransform,
        input_cls: Type[Input] = IceVisionInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "ObjectDetectionData":
        """.. _COCO: https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch.

        Creates a :class:`~flash.image.detection.data.ObjectDetectionData` object from the given data folders
        and annotation files in the `COCO JSON format <https://cocodataset.org/#format-data>`_.

        For help understanding and using the COCO format, take a look at this tutorial: `Create COCO annotations from
        scratch <COCO>`__.

        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_folder: The folder containing images to use when training.
            train_ann_file: The COCO format annotation file to use when training.
            val_folder: The folder containing images to use when validating.
            val_ann_file: The COCO format annotation file to use when validating.
            test_folder: The folder containing images to use when testing.
            test_ann_file: The COCO format annotation file to use when testing.
            predict_folder: The folder containing images to use when predicting.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
              :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.image.detection.data.ObjectDetectionData`.

        Examples
        ________

        .. testsetup::

            >>> import os
            >>> import json
            >>> import numpy as np
            >>> from PIL import Image
            >>> rand_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))
            >>> os.makedirs("train_folder", exist_ok=True)
            >>> os.makedirs("predict_folder", exist_ok=True)
            >>> _ = [rand_image.save(os.path.join("train_folder", f"image_{i}.png")) for i in range(1, 4)]
            >>> _ = [rand_image.save(os.path.join("predict_folder", f"predict_image_{i}.png")) for i in range(1, 4)]
            >>> annotations = {"annotations": [
            ...     {"area": 50, "bbox": [10, 20, 5, 10], "category_id": 1, "id": 1, "image_id": 1, "iscrowd": 0},
            ...     {"area": 100, "bbox": [20, 30, 10, 10], "category_id": 2, "id": 2, "image_id": 2, "iscrowd": 0},
            ...     {"area": 125, "bbox": [10, 20, 5, 25], "category_id": 1, "id": 3, "image_id": 3, "iscrowd": 0},
            ... ], "categories": [
            ...     {"id": 1, "name": "cat", "supercategory": "cat"},
            ...     {"id": 2, "name": "dog", "supercategory": "dog"},
            ... ], "images": [
            ...     {"file_name": "image_1.png", "height": 64, "width": 64, "id": 1},
            ...     {"file_name": "image_2.png", "height": 64, "width": 64, "id": 2},
            ...     {"file_name": "image_3.png", "height": 64, "width": 64, "id": 3},
            ... ]}
            >>> with open("train_annotations.json", "w") as annotation_file:
            ...     json.dump(annotations, annotation_file)

        The folder ``train_folder`` has the following contents:

        .. code-block::

            train_folder
            ├── image_1.png
            ├── image_2.png
            ├── image_3.png
            ...

        The file ``train_annotations.json`` contains the following:

        .. code-block::

            {
                "annotations": [
                    {"area": 50, "bbox": [10, 20, 5, 10], "category_id": 1, "id": 1, "image_id": 1, "iscrowd": 0},
                    {"area": 100, "bbox": [20, 30, 10, 10], "category_id": 2, "id": 2, "image_id": 2, "iscrowd": 0},
                    {"area": 125, "bbox": [10, 20, 5, 25], "category_id": 1, "id": 3, "image_id": 3, "iscrowd": 0}
                ], "categories": [
                    {"id": 1, "name": "cat", "supercategory": "cat"},
                    {"id": 2, "name": "dog", "supercategory": "dog"}
                ], "images": [
                    {"file_name": "image_1.png", "height": 64, "width": 64, "id": 1},
                    {"file_name": "image_2.png", "height": 64, "width": 64, "id": 2},
                    {"file_name": "image_3.png", "height": 64, "width": 64, "id": 3}
                ]
            }

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.image import ObjectDetector, ObjectDetectionData
            >>> datamodule = ObjectDetectionData.from_coco(
            ...     train_folder="train_folder",
            ...     train_ann_file="train_annotations.json",
            ...     predict_folder="predict_folder",
            ...     transform_kwargs=dict(image_size=(128, 128)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            3
            >>> datamodule.labels
            ['background', 'cat', 'dog']
            >>> model = ObjectDetector(num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> import shutil
            >>> shutil.rmtree("train_folder")
            >>> shutil.rmtree("predict_folder")
            >>> os.remove("train_annotations.json")
        """
        return cls.from_icedata(
            train_folder=train_folder,
            train_ann_file=train_ann_file,
            val_folder=val_folder,
            val_ann_file=val_ann_file,
            test_folder=test_folder,
            test_ann_file=test_ann_file,
            predict_folder=predict_folder,
            parser=COCOBBoxParser,
            input_cls=input_cls,
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_voc(
        cls,
        labels: List[str],
        train_folder: Optional[str] = None,
        train_ann_folder: Optional[str] = None,
        val_folder: Optional[str] = None,
        val_ann_folder: Optional[str] = None,
        test_folder: Optional[str] = None,
        test_ann_folder: Optional[str] = None,
        predict_folder: Optional[str] = None,
        transform: INPUT_TRANSFORM_TYPE = IceVisionInputTransform,
        input_cls: Type[Input] = IceVisionInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "ObjectDetectionData":
        """.. _PASCAL: http://host.robots.ox.ac.uk/pascal/VOC/

        Creates a :class:`~flash.image.detection.data.ObjectDetectionData` object from the given data folders
        and annotation files in the `PASCAL VOC (Visual Object Challenge) XML format <PASCAL>`_.

        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            labels: A list of class labels. Note that the list should not include a label for the background class which
                will be added automatically as class zero (additional labels will be sorted).
            train_folder: The folder containing images to use when training.
            train_ann_folder: The folder containing VOC format annotation files to use when training.
            val_folder: The folder containing images to use when validating.
            val_ann_folder: The folder containing VOC format annotation files to use when validating.
            test_folder: The folder containing images to use when testing.
            test_ann_folder: The folder containing VOC format annotation files to use when testing.
            predict_folder: The folder containing images to use when predicting.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
              :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.image.detection.data.ObjectDetectionData`.

        Examples
        ________

        .. testsetup::

            >>> import os
            >>> import numpy as np
            >>> from PIL import Image
            >>> rand_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))
            >>> os.makedirs("train_folder", exist_ok=True)
            >>> os.makedirs("train_annotations", exist_ok=True)
            >>> os.makedirs("predict_folder", exist_ok=True)
            >>> _ = [rand_image.save(os.path.join("train_folder", f"image_{i}.png")) for i in range(1, 4)]
            >>> _ = [rand_image.save(os.path.join("predict_folder", f"predict_image_{i}.png")) for i in range(1, 4)]
            >>> bboxes = [[10, 20, 15, 30], [20, 30, 30, 40], [10, 20, 15, 45]]
            >>> labels = ["cat", "dog", "cat"]
            >>> for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            ...     xmin, ymin, xmax, ymax = bbox
            ...     annotation = (
            ...         f"<annotation><filename>image_{i + 1}.png</filename>"
            ...         f"<path>image_{i + 1}.png</path><source><database>example</database></source>"
            ...         "<size><width>64</width><height>64</height><depth>3</depth></size>"
            ...         f"<object><name>{label}</name><pose>Unspecified</pose><truncated>0</truncated>"
            ...         f"<difficult>0</difficult><occluded>0</occluded><bndbox><xmin>{xmin}</xmin><xmax>{xmax}</xmax>"
            ...         f"<ymin>{ymin}</ymin><ymax>{ymax}</ymax></bndbox></object></annotation>"
            ...     )
            ...     with open(os.path.join("train_annotations", f"image_{i+1}.xml"), "w") as file:
            ...         _ = file.write(annotation)

        The folder ``train_folder`` has the following contents:

        .. code-block::

            train_folder
            ├── image_1.png
            ├── image_2.png
            ├── image_3.png
            ...

        The folder ``train_annotations`` has the following contents:

        .. code-block::

            train_annotations
            ├── image_1.xml
            ├── image_2.xml
            ├── image_3.xml
            ...

        The file ``image_1.xml`` contains the following:

        .. code-block::

            <annotation>
                <filename>image_0.png</filename>
                <path>image_0.png</path>
                <source><database>example</database></source>
                <size><width>64</width><height>64</height><depth>3</depth></size>
                <object>
                    <name>cat</name>
                    <pose>Unspecified</pose>
                    <truncated>0</truncated>
                    <difficult>0</difficult>
                    <occluded>0</occluded>
                    <bndbox><xmin>10</xmin><xmax>15</xmax><ymin>20</ymin><ymax>30</ymax></bndbox>
                </object>
            </annotation>

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.image import ObjectDetector, ObjectDetectionData
            >>> datamodule = ObjectDetectionData.from_voc(
            ...     ["cat", "dog"],
            ...     train_folder="train_folder",
            ...     train_ann_folder="train_annotations",
            ...     predict_folder="predict_folder",
            ...     transform_kwargs=dict(image_size=(128, 128)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            3
            >>> datamodule.labels
            ['background', 'cat', 'dog']
            >>> model = ObjectDetector(num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> import shutil
            >>> shutil.rmtree("train_folder")
            >>> shutil.rmtree("predict_folder")
            >>> shutil.rmtree("train_annotations")
        """
        return cls.from_icedata(
            train_folder=train_folder,
            train_ann_file=train_ann_folder,
            val_folder=val_folder,
            val_ann_file=val_ann_folder,
            test_folder=test_folder,
            test_ann_file=test_ann_folder,
            predict_folder=predict_folder,
            parser=partial(VOCBBoxParser, class_map=ClassMap(list(sorted_alphanumeric(labels)))),
            input_cls=input_cls,
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_via(
        cls,
        labels: List[str],
        label_field: str = "label",
        train_folder: Optional[str] = None,
        train_ann_file: Optional[str] = None,
        val_folder: Optional[str] = None,
        val_ann_file: Optional[str] = None,
        test_folder: Optional[str] = None,
        test_ann_file: Optional[str] = None,
        predict_folder: Optional[str] = None,
        transform: INPUT_TRANSFORM_TYPE = IceVisionInputTransform,
        input_cls: Type[Input] = IceVisionInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "ObjectDetectionData":
        """Creates a :class:`~flash.image.detection.data.ObjectDetectionData` object from the given data folders
        and annotation files in the VIA (`VGG Image Annotator <https://www.robots.ox.ac.uk/~vgg/software/via/>`_)
        `JSON format <https://gitlab.com/vgg/via/-/blob/via-3.x.y/via-3.x.y/CodeDoc.md#structure-of-via-project-
        json-file>`_.

        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            labels: A list of class labels. Not that the list should not include a label for the background class which
                will be added automatically as class zero (additional labels will be sorted).
            label_field: The field within ``region_attributes`` which corresponds to the region label.
            train_folder: The folder containing images to use when training.
            train_ann_file: The VIA format annotation file to use when training.
            val_folder: The folder containing images to use when validating.
            val_ann_file: The VIA format annotation file to use when validating.
            test_folder: The folder containing images to use when testing.
            test_ann_file: The VIA format annotation file to use when testing.
            predict_folder: The folder containing images to use when predicting.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
              :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.image.detection.data.ObjectDetectionData`.

        Examples
        ________

        .. testsetup::

            >>> import os
            >>> import json
            >>> import numpy as np
            >>> from PIL import Image
            >>> rand_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))
            >>> os.makedirs("train_folder", exist_ok=True)
            >>> os.makedirs("predict_folder", exist_ok=True)
            >>> _ = [rand_image.save(os.path.join("train_folder", f"image_{i}.png")) for i in range(1, 4)]
            >>> _ = [rand_image.save(os.path.join("predict_folder", f"predict_image_{i}.png")) for i in range(1, 4)]
            >>> annotations = {
            ...     f"image_{i+1}.png": {
            ...         "filename": f"image_{i+1}.png",
            ...         "regions": [{
            ...                 "shape_attributes": {"name": "rect", "x": 10, "y": 20, "width": 5, "height": 10},
            ...                 "region_attributes": {"label": lb},
            ...         }]
            ...     } for i, lb in enumerate(["cat", "dog", "cat"])
            ... }
            >>> with open("train_annotations.json", "w") as annotation_file:
            ...     json.dump(annotations, annotation_file)

        The folder ``train_folder`` has the following contents:

        .. code-block::

            train_folder
            ├── image_1.png
            ├── image_2.png
            ├── image_3.png
            ...

        The file ``train_annotations.json`` contains the following:

        .. code-block::

            {
                "image_1.png": {
                    "filename": "image_1.png",
                    "regions": [{
                        "shape_attributes": {"name": "rect", "x": 10, "y": 20, "width": 5, "height": 10},
                        "region_attributes": {"label": "cat"}
                    }]
                }, "image_2.png": {
                    "filename": "image_2.png",
                    "regions": [{
                        "shape_attributes": {"name": "rect", "x": 20, "y": 30, "width": 10, "height": 10},
                        "region_attributes": {"label": "dog"}}
                ]}, "image_3.png": {
                    "filename": "image_3.png",
                    "regions": [{
                        "shape_attributes": {"name": "rect", "x": 10, "y": 20, "width": 5, "height": 25},
                        "region_attributes": {"label": "cat"}
                    }]
                }
            }

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.image import ObjectDetector, ObjectDetectionData
            >>> datamodule = ObjectDetectionData.from_via(
            ...     ["cat", "dog"],
            ...     train_folder="train_folder",
            ...     train_ann_file="train_annotations.json",
            ...     predict_folder="predict_folder",
            ...     transform_kwargs=dict(image_size=(128, 128)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            3
            >>> datamodule.labels
            ['background', 'cat', 'dog']
            >>> model = ObjectDetector(num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> import shutil
            >>> shutil.rmtree("train_folder")
            >>> shutil.rmtree("predict_folder")
            >>> os.remove("train_annotations.json")
        """
        return cls.from_icedata(
            train_folder=train_folder,
            train_ann_file=train_ann_file,
            val_folder=val_folder,
            val_ann_file=val_ann_file,
            test_folder=test_folder,
            test_ann_file=test_ann_file,
            predict_folder=predict_folder,
            parser=partial(
                VIABBoxParser,
                class_map=ClassMap(list(sorted_alphanumeric(labels))),
                label_field=label_field,
            ),
            input_cls=input_cls,
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
        iscrowd: str = "iscrowd",
        transform: INPUT_TRANSFORM_TYPE = IceVisionInputTransform,
        input_cls: Type[Input] = ObjectDetectionFiftyOneInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "ObjectDetectionData":
        """Load the :class:`~flash.image.detection.data.ObjectDetectionData` from FiftyOne ``SampleCollection``
        objects.

        Targets will be extracted from the ``label_field`` in the ``SampleCollection`` objects.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_dataset: The ``SampleCollection`` to use when training.
            val_dataset: The ``SampleCollection`` to use when validating.
            test_dataset: The ``SampleCollection`` to use when testing.
            predict_dataset: The ``SampleCollection`` to use when predicting.
            label_field: The field in the ``SampleCollection`` objects containing the targets.
            iscrowd: The field in the ``SampleCollection`` objects containing the ``iscrowd`` annotation (if required).
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
              :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.image.detection.data.ObjectDetectionData`.

        Examples
        ________

        .. testsetup::

            >>> import numpy as np
            >>> from PIL import Image
            >>> rand_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))
            >>> _ = [rand_image.save(f"image_{i}.png") for i in range(1, 4)]
            >>> _ = [rand_image.save(f"predict_image_{i}.png") for i in range(1, 4)]

        .. doctest::

            >>> import numpy as np
            >>> import fiftyone as fo
            >>> from flash import Trainer
            >>> from flash.image import ObjectDetector, ObjectDetectionData
            >>> train_dataset = fo.Dataset.from_images(
            ...     ["image_1.png", "image_2.png", "image_3.png"]
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            <BLANKLINE>
            ...
            >>> samples = [train_dataset[filepath] for filepath in train_dataset.values("filepath")]
            >>> for sample, label, bounding_box in zip(
            ...     samples,
            ...     ["cat", "dog", "cat"],
            ...     [[0.1, 0.2, 0.15, 0.3], [0.2, 0.3, 0.3, 0.4], [0.1, 0.2, 0.15, 0.45]],
            ... ):
            ...     sample["ground_truth"] = fo.Detections(
            ...         detections=[fo.Detection(label=label, bounding_box=bounding_box)],
            ...     )
            ...     sample.save()
            ...
            >>> predict_dataset = fo.Dataset.from_images(
            ...     ["predict_image_1.png", "predict_image_2.png", "predict_image_3.png"]
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            <BLANKLINE>
            ...
            >>> datamodule = ObjectDetectionData.from_fiftyone(
            ...     train_dataset=train_dataset,
            ...     predict_dataset=predict_dataset,
            ...     transform_kwargs=dict(image_size=(128, 128)),
            ...     batch_size=2,
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            <BLANKLINE>
            ...
            >>> datamodule.num_classes
            3
            >>> datamodule.labels
            ['background', 'cat', 'dog']
            >>> model = ObjectDetector(num_classes=datamodule.num_classes)
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

        ds_kw = dict()

        return cls(
            input_cls(RunningStage.TRAINING, train_dataset, label_field, iscrowd, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_dataset, label_field, iscrowd, **ds_kw),
            input_cls(RunningStage.TESTING, test_dataset, label_field, iscrowd, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_dataset, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_folders(
        cls,
        predict_folder: Optional[str] = None,
        predict_transform: INPUT_TRANSFORM_TYPE = IceVisionInputTransform,
        input_cls: Type[Input] = IceVisionInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "DataModule":
        """Creates a :class:`~flash.image.detection.data.ObjectDetectionData` object from the given data folders
        This is currently support only for the predicting stage.

        Args:
            predict_folder: The folder containing the predict data.
            predict_transform: The dictionary of transforms to use during predicting which maps
            data_module_kwargs: The keywords arguments for creating the datamodule.

        Returns:
            The constructed data module.
        """
        return cls(
            predict_input=input_cls(RunningStage.PREDICTING, predict_folder),
            transform=predict_transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )
