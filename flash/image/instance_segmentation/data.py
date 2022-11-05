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
from typing import Any, Callable, Dict, List, Optional, Type, Union

from torch import tensor

from flash.core.data.data_module import DataModule
from flash.core.data.io.input import DataKeys, Input
from flash.core.data.io.output_transform import OutputTransform
from flash.core.data.utilities.sort import sorted_alphanumeric
from flash.core.integrations.icevision.data import IceVisionInput
from flash.core.integrations.icevision.transforms import IceVisionInputTransform
from flash.core.utilities.imports import (
    _ICEVISION_AVAILABLE,
    _IMAGE_EXTRAS_TESTING,
    _TORCHVISION_AVAILABLE,
    _TORCHVISION_GREATER_EQUAL_0_9,
)
from flash.core.utilities.stages import RunningStage
from flash.core.utilities.types import INPUT_TRANSFORM_TYPE

if _ICEVISION_AVAILABLE:
    from icevision.core import ClassMap
    from icevision.parsers import COCOMaskParser, Parser, VOCMaskParser
else:
    COCOMaskParser = object
    VOCMaskParser = object
    Parser = object

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as T

    if _TORCHVISION_GREATER_EQUAL_0_9:
        from torchvision.transforms import InterpolationMode
    else:

        class InterpolationMode:
            NEAREST = "nearest"


# Skip doctests if requirements aren't available
if not _IMAGE_EXTRAS_TESTING:
    __doctest_skip__ = ["InstanceSegmentationData", "InstanceSegmentationData.*"]


class InstanceSegmentationOutputTransform(OutputTransform):
    def per_sample_transform(self, sample: Any) -> Any:
        resize = T.Resize(sample[DataKeys.METADATA]["size"], interpolation=InterpolationMode.NEAREST)
        sample[DataKeys.PREDS]["masks"] = [resize(tensor(mask)) for mask in sample[DataKeys.PREDS]["masks"]]
        return sample[DataKeys.PREDS]


class InstanceSegmentationData(DataModule):

    input_transform_cls = IceVisionInputTransform
    output_transform_cls = InstanceSegmentationOutputTransform

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
        parser: Optional[Union[Callable, Type[Parser]]] = None,
        input_cls: Type[Input] = IceVisionInput,
        transform: INPUT_TRANSFORM_TYPE = IceVisionInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs,
    ) -> "InstanceSegmentationData":

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
        input_cls: Type[Input] = IceVisionInput,
        transform: INPUT_TRANSFORM_TYPE = IceVisionInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ):
        """Creates a :class:`~flash.image.instance_segmentation.data.InstanceSegmentationData` object from the
        given data folders and annotation files in the `COCO JSON format <https://cocodataset.org/#format-data>`_.

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
            The constructed :class:`~flash.image.instance_segmentation.data.InstanceSegmentationData`.

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
            ...     {"area": 50, "bbox": [10, 20, 5, 10], "category_id": 1, "id": 1, "image_id": 1, "iscrowd": 0,
            ...     "segmentation": [[10, 20, 15, 20, 15, 30, 10, 30]]},
            ...     {"area": 100, "bbox": [20, 30, 10, 10], "category_id": 2, "id": 2, "image_id": 2, "iscrowd": 0,
            ...     "segmentation": [[20, 30, 30, 30, 30, 40, 20, 40]]},
            ...     {"area": 125, "bbox": [10, 20, 5, 25], "category_id": 1, "id": 3, "image_id": 3, "iscrowd": 0,
            ...     "segmentation": [[10, 20, 15, 20, 15, 45, 10, 45]]},
            ... ], "categories": [
            ...     {"id": 1, "name": "cat", "supercategory": "annimal"},
            ...     {"id": 2, "name": "dog", "supercategory": "annimal"},
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
                    {"area": 50, "bbox": [10, 20, 5, 10], "category_id": 1, "id": 1, "image_id": 1, "iscrowd": 0,
                    "segmentation": [[10, 20, 15, 20, 15, 30, 10, 30]]},
                    {"area": 100, "bbox": [20, 30, 10, 10], "category_id": 2, "id": 2, "image_id": 2, "iscrowd": 0,
                    "segmentation": [[20, 30, 30, 30, 30, 40, 20, 40]]},
                    {"area": 125, "bbox": [10, 20, 5, 25], "category_id": 1, "id": 3, "image_id": 3, "iscrowd": 0,
                    "segmentation": [[10, 20, 15, 20, 15, 45, 10, 45]]}
                ], "categories": [
                    {"id": 1, "name": "cat", "supercategory": "annimal"},
                    {"id": 2, "name": "dog", "supercategory": "annimal"}
                ], "images": [
                    {"file_name": "image_1.png", "height": 64, "width": 64, "id": 1},
                    {"file_name": "image_2.png", "height": 64, "width": 64, "id": 2},
                    {"file_name": "image_3.png", "height": 64, "width": 64, "id": 3}
                ]
            }

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.image import InstanceSegmentation, InstanceSegmentationData
            >>> datamodule = InstanceSegmentationData.from_coco(
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
            >>> model = InstanceSegmentation(num_classes=datamodule.num_classes)
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
            parser=COCOMaskParser,
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
        train_target_folder: Optional[str] = None,
        train_ann_folder: Optional[str] = None,
        val_folder: Optional[str] = None,
        val_target_folder: Optional[str] = None,
        val_ann_folder: Optional[str] = None,
        test_folder: Optional[str] = None,
        test_target_folder: Optional[str] = None,
        test_ann_folder: Optional[str] = None,
        predict_folder: Optional[str] = None,
        input_cls: Type[Input] = IceVisionInput,
        transform: INPUT_TRANSFORM_TYPE = IceVisionInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ):
        """Creates a :class:`~flash.image.instance_segmentation.data.InstanceSegmentationData` object from the
        given data folders, mask folders, and annotation files in the `PASCAL VOC (Visual Object Challenge) XML
        format <PASCAL>`_.

        .. note:: All three arguments `*_folder`, `*_target_folder`, and `*_ann_folder` are needed to load data for a
            particular stage.

        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            labels: A list of class labels. Note that the list should not include a label for the background class which
                will be added automatically as class zero (additional labels will be sorted).
            train_folder: The folder containing images to use when training.
            train_target_folder: The folder containing mask images to use when training.
            train_ann_folder: The folder containing VOC format annotation files to use when training.
            val_folder: The folder containing images to use when validating.
            val_target_folder: The folder containing mask images to use when validating.
            val_ann_folder: The folder containing VOC format annotation files to use when validating.
            test_folder: The folder containing images to use when testing.
            test_target_folder: The folder containing mask images to use when testing.
            test_ann_folder: The folder containing VOC format annotation files to use when testing.
            predict_folder: The folder containing images to use when predicting.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
              :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.image.instance_segmentation.data.InstanceSegmentationData`.

        Examples
        ________

        .. testsetup::

            >>> import os
            >>> import numpy as np
            >>> from PIL import Image
            >>> rand_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))
            >>> rand_mask = Image.fromarray(np.random.randint(0, 3, (64, 64), dtype="uint8"))
            >>> os.makedirs("train_folder", exist_ok=True)
            >>> os.makedirs("train_masks", exist_ok=True)
            >>> os.makedirs("train_annotations", exist_ok=True)
            >>> os.makedirs("predict_folder", exist_ok=True)
            >>> _ = [rand_image.save(os.path.join("train_folder", f"image_{i}.png")) for i in range(1, 4)]
            >>> _ = [rand_mask.save(os.path.join("train_masks", f"image_{i}.png")) for i in range(1, 4)]
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
            ...     with open(os.path.join("train_annotations", f"image_{i + 1}.xml"), "w") as file:
            ...         _ = file.write(annotation)

        The folder ``train_folder`` has the following contents:

        .. code-block::

            train_folder
            ├── image_1.png
            ├── image_2.png
            ├── image_3.png
            ...

        The folder ``train_masks`` has the following contents:

        .. code-block::

            train_masks
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
            >>> from flash.image import InstanceSegmentation, InstanceSegmentationData
            >>> datamodule = InstanceSegmentationData.from_voc(
            ...     ["cat", "dog"],
            ...     train_folder="train_folder",
            ...     train_target_folder="train_masks",
            ...     train_ann_folder="train_annotations",
            ...     predict_folder="predict_folder",
            ...     transform_kwargs=dict(image_size=(128, 128)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            3
            >>> datamodule.labels
            ['background', 'cat', 'dog']
            >>> model = InstanceSegmentation(num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> import shutil
            >>> shutil.rmtree("train_folder")
            >>> shutil.rmtree("train_masks")
            >>> shutil.rmtree("predict_folder")
            >>> shutil.rmtree("train_annotations")
        """
        return cls.from_icedata(
            train_folder=train_folder,
            train_ann_file=train_ann_folder,
            train_parser_kwargs={"masks_dir": train_target_folder},
            val_folder=val_folder,
            val_ann_file=val_ann_folder,
            val_parser_kwargs={"masks_dir": val_target_folder},
            test_folder=test_folder,
            test_ann_file=test_ann_folder,
            test_parser_kwargs={"masks_dir": test_target_folder},
            predict_folder=predict_folder,
            parser=partial(VOCMaskParser, class_map=ClassMap(list(sorted_alphanumeric(labels)))),
            input_cls=input_cls,
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
        """Creates a :class:`~flash.core.data.data_module.DataModule` object from the given folders.

        This is supported only for the predicting stage.

        Args:
            predict_folder: The folder containing the predict data.
            predict_transform: The dictionary of transforms to use during predicting which maps.
            input_cls: The :class:`~flash.core.data.io.input.Input` used to create the dataset.
            transform_kwargs: Keyword arguments provided to the transform on instantiation.
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

    @classmethod
    def from_files(
        cls,
        predict_files: Optional[List[str]] = None,
        predict_transform: INPUT_TRANSFORM_TYPE = IceVisionInputTransform,
        input_cls: Type[Input] = IceVisionInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "DataModule":
        """Creates a :class:`~flash.core.data.data_module.DataModule` object from the given a list of files.

        This is supported only for the predicting stage.

        Args:
            predict_files: The list of files containing the predict data.
            predict_transform: The dictionary of transforms to use during predicting which maps.
            input_cls: The :class:`~flash.core.data.io.input.Input` used to create the dataset.
            transform_kwargs: Keyword arguments provided to the transform on instantiation.
            data_module_kwargs: The keywords arguments for creating the datamodule.

        Returns:
            The constructed data module.
        """
        return cls(
            predict_input=input_cls(RunningStage.PREDICTING, predict_files),
            transform=predict_transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )
