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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

from flash.core.data.data_module import DataModule
from flash.core.data.io.input import Input
from flash.core.integrations.icevision.data import IceVisionInput
from flash.core.utilities.imports import _ICEVISION_AVAILABLE, _IMAGE_EXTRAS_TESTING
from flash.core.utilities.stages import RunningStage
from flash.core.utilities.types import INPUT_TRANSFORM_TYPE
from flash.image.keypoint_detection.input_transform import KeypointDetectionInputTransform

if _ICEVISION_AVAILABLE:
    from icevision.core import KeyPoints, KeypointsMetadata
    from icevision.parsers import COCOKeyPointsParser, Parser
else:
    COCOKeyPointsParser = object
    Parser = object
    KeyPoints = object


# Skip doctests if requirements aren't available
if not _IMAGE_EXTRAS_TESTING:
    __doctest_skip__ = ["KeypointDetectionData", "KeypointDetectionData.*"]


class FlashCOCOKeyPointsParser(COCOKeyPointsParser):
    def __init__(
        self,
        annotations_filepath: Union[str, Path, dict],
        img_dir: Union[str, Path],
    ):
        super().__init__(annotations_filepath, img_dir)

        categories = self.annotations_dict["categories"]
        self.keypoint_labels = categories[0]["keypoints"]
        for o in categories[1:]:
            if not o["keypoints"] == self.keypoint_labels:
                raise ValueError(
                    "When performing keypoint detection with multiple categories, all categories are expected to have "
                    f"the same keypoints. Found {self.keypoint_labels} for category with ID {categories[0]['id']} and "
                    f"{o['keypoints']} for category with ID {o['id']}."
                )

    def keypoints(self, o) -> List[KeyPoints]:
        meta = KeypointsMetadata()
        meta.labels = self.keypoint_labels
        return [KeyPoints.from_xyv(o["keypoints"], meta)] if sum(o["keypoints"]) > 0 else []


class KeypointDetectionData(DataModule):
    """The ``KeypointDetectionData`` class is a :class:`~flash.core.data.data_module.DataModule` with a set of
    classmethods for loading data for keypoint detection."""

    input_transform_cls = KeypointDetectionInputTransform

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
        transform: INPUT_TRANSFORM_TYPE = KeypointDetectionInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "KeypointDetectionData":

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
        transform: INPUT_TRANSFORM_TYPE = KeypointDetectionInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ):
        """Creates a :class:`~flash.image.keypoint_detection.data.KeypointDetectionData` object from the given data
        folders and annotation files in the `COCO JSON format <https://cocodataset.org/#format-data>`_.

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
            The constructed :class:`~flash.image.keypoint_detection.data.KeypointDetectionData`.

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
            ...     {"area": 50, "bbox": [10, 20, 5, 10], "num_keypoints": 2, "keypoints": [10, 15, 2, 20, 30, 2],
            ...     "category_id": 1, "id": 1, "image_id": 1, "iscrowd": 0},
            ...     {"area": 100, "bbox": [20, 30, 10, 10], "num_keypoints": 2, "keypoints": [20, 30, 2, 30, 40, 2],
            ...     "category_id": 2, "id": 2, "image_id": 2, "iscrowd": 0},
            ...     {"area": 125, "bbox": [10, 20, 5, 25], "num_keypoints": 2, "keypoints": [10, 15, 2, 20, 45, 2],
            ...     "category_id": 1, "id": 3, "image_id": 3, "iscrowd": 0},
            ... ], "categories": [
            ...     {"id": 1, "name": "cat", "supercategory": "cat", "keypoints": ["left ear", "right ear"]},
            ...     {"id": 2, "name": "dog", "supercategory": "dog", "keypoints": ["left ear", "right ear"]},
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
                    {
                        "area": 50, "bbox": [10, 20, 5, 10], "num_keypoints": 2, "keypoints": [10, 15, 2, 20, 30, 2],
                        "category_id": 1, "id": 1, "image_id": 1, "iscrowd": 0
                    }, {
                        "area": 100, "bbox": [20, 30, 10, 10], "num_keypoints": 2, "keypoints": [20, 30, 2, 30, 40, 2],
                        "category_id": 2, "id": 2, "image_id": 2, "iscrowd": 0
                    }, {
                        "area": 125, "bbox": [10, 20, 5, 25], "num_keypoints": 2, "keypoints": [10, 15, 2, 20, 45, 2],
                        "category_id": 1, "id": 3, "image_id": 3, "iscrowd": 0
                    }
                ], "categories": [
                    {"id": 1, "name": "cat", "supercategory": "cat", "keypoints": ["left ear", "right ear"]},
                    {"id": 2, "name": "dog", "supercategory": "dog", "keypoints": ["left ear", "right ear"]}
                ], "images": [
                    {"file_name": "image_1.png", "height": 64, "width": 64, "id": 1},
                    {"file_name": "image_2.png", "height": 64, "width": 64, "id": 2},
                    {"file_name": "image_3.png", "height": 64, "width": 64, "id": 3}
                ]
            }

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.image import KeypointDetector, KeypointDetectionData
            >>> datamodule = KeypointDetectionData.from_coco(
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
            >>> model = KeypointDetector(2, num_classes=datamodule.num_classes)
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
            parser=FlashCOCOKeyPointsParser,
            input_cls=input_cls,
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_folders(
        cls,
        predict_folder: Optional[str] = None,
        input_cls: Type[Input] = IceVisionInput,
        predict_transform: INPUT_TRANSFORM_TYPE = KeypointDetectionInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "DataModule":
        """Creates a :class:`~flash.core.data.data_module.DataModule` object from the given folders.

        This is supported only for the predicting stage.

        Args:
            predict_folder: The folder containing the predict data.
            input_cls: The :class:`~flash.core.data.io.input.Input` used to create the dataset.
            predict_transform: The dictionary of transforms to use during predicting which maps
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
        input_cls: Type[Input] = IceVisionInput,
        predict_transform: INPUT_TRANSFORM_TYPE = KeypointDetectionInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "DataModule":
        """Creates a :class:`~flash.core.data.data_module.DataModule` object from the given a list of files.

        This is supported only for the predicting stage.

        Args:
            predict_files: The list of files containing the predict data.
            input_cls: The :class:`~flash.core.data.io.input.Input` used to create the dataset.
            predict_transform: The dictionary of transforms to use during predicting which maps.
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
