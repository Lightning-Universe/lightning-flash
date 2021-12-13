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
from typing import Any, Dict, List, Optional, Type

from flash.core.data.data_module import DataModule
from flash.core.data.data_pipeline import DataPipelineState
from flash.core.data.io.input import Input
from flash.core.integrations.icevision.data import IceVisionInput
from flash.core.integrations.icevision.transforms import IceVisionInputTransform
from flash.core.utilities.imports import _ICEVISION_AVAILABLE
from flash.core.utilities.stages import RunningStage
from flash.core.utilities.types import INPUT_TRANSFORM_TYPE

if _ICEVISION_AVAILABLE:
    from icevision.parsers import COCOKeyPointsParser, Parser
else:
    COCOKeyPointsParser = object
    Parser = object


class KeypointDetectionData(DataModule):

    input_transform_cls = IceVisionInputTransform

    @classmethod
    def from_icedata(
        cls,
        train_folder: Optional[str] = None,
        train_ann_file: Optional[str] = None,
        val_folder: Optional[str] = None,
        val_ann_file: Optional[str] = None,
        test_folder: Optional[str] = None,
        test_ann_file: Optional[str] = None,
        predict_folder: Optional[str] = None,
        train_transform: INPUT_TRANSFORM_TYPE = IceVisionInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = IceVisionInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = IceVisionInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = IceVisionInputTransform,
        parser: Optional[Type[Parser]] = Parser,
        input_cls: Type[Input] = IceVisionInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "KeypointDetectionData":

        ds_kw = dict(parser=parser, data_pipeline_state=DataPipelineState(), transform_kwargs=transform_kwargs)

        return cls(
            input_cls(RunningStage.TRAINING, train_folder, train_ann_file, transform=train_transform, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_folder, val_ann_file, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_folder, test_ann_file, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_folder, transform=predict_transform, **ds_kw),
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
        train_transform: INPUT_TRANSFORM_TYPE = IceVisionInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = IceVisionInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = IceVisionInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = IceVisionInputTransform,
        parser: Optional[Type[Parser]] = COCOKeyPointsParser,
        input_cls: Type[Input] = IceVisionInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ):
        """Creates a :class:`~flash.image.keypoint_detection.data.KeypointDetectionData` object from the given data
        folders and annotation files in the COCO format.

        Args:
            train_folder: The folder containing the train data.
            train_ann_file: The COCO format annotation file.
            val_folder: The folder containing the validation data.
            val_ann_file: The COCO format annotation file.
            test_folder: The folder containing the test data.
            test_ann_file: The COCO format annotation file.
            predict_folder: The folder containing the predict data.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            input_cls: The :class:`~flash.core.data.io.input.Input` used to create the dataset.
            transform_kwargs: Keyword arguments provided to the transform on instantiation.
            data_module_kwargs: Keyword arguments provided to the DataModule on instantiation.
        """
        ds_kw = dict(parser=parser, data_pipeline_state=DataPipelineState(), transform_kwargs=transform_kwargs)

        return cls(
            input_cls(RunningStage.TRAINING, train_folder, train_ann_file, transform=train_transform, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_folder, val_ann_file, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_folder, test_ann_file, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_folder, transform=predict_transform, **ds_kw),
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
            predict_transform: The dictionary of transforms to use during predicting which maps
            input_cls: The :class:`~flash.core.data.io.input.Input` used to create the dataset.
            transform_kwargs: Keyword arguments provided to the transform on instantiation.
            data_module_kwargs: The keywords arguments for creating the datamodule.

        Returns:
            The constructed data module.
        """
        ds_kw = dict(transform=predict_transform, transform_kwargs=transform_kwargs)

        return cls(
            predict_input=input_cls(RunningStage.PREDICTING, predict_folder, **ds_kw),
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
        ds_kw = dict(transform=predict_transform, transform_kwargs=transform_kwargs)

        return cls(
            predict_input=input_cls(RunningStage.PREDICTING, predict_files, **ds_kw),
            **data_module_kwargs,
        )

    from_tensor = None
    from_json = None
    from_csv = None
    from_datasets = None
