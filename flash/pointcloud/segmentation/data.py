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

from torch.utils.data import Dataset

from flash.core.data.data_module import DataModule
from flash.core.data.io.input import Input
from flash.core.data.io.input_transform import InputTransform
from flash.core.utilities.stability import beta
from flash.core.utilities.stages import RunningStage
from flash.core.utilities.types import INPUT_TRANSFORM_TYPE
from flash.pointcloud.segmentation.input import PointCloudSegmentationDatasetInput, PointCloudSegmentationFoldersInput


@beta("Point cloud segmentation is currently in Beta.")
class PointCloudSegmentationData(DataModule):

    input_transform_cls = InputTransform

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[str] = None,
        val_folder: Optional[str] = None,
        test_folder: Optional[str] = None,
        predict_folder: Optional[str] = None,
        input_cls: Type[Input] = PointCloudSegmentationFoldersInput,
        transform: INPUT_TRANSFORM_TYPE = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "PointCloudSegmentationData":

        ds_kw = dict()

        return cls(
            input_cls(RunningStage.TRAINING, train_folder, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_folder, **ds_kw),
            input_cls(RunningStage.TESTING, test_folder, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_folder, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_files(
        cls,
        predict_files: Optional[List[str]] = None,
        input_cls: Type[Input] = PointCloudSegmentationFoldersInput,
        transform: INPUT_TRANSFORM_TYPE = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "PointCloudSegmentationData":

        ds_kw = dict()

        return cls(
            predict_input=input_cls(RunningStage.PREDICTING, predict_files, **ds_kw),
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
        input_cls: Type[Input] = PointCloudSegmentationDatasetInput,
        transform: INPUT_TRANSFORM_TYPE = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "PointCloudSegmentationData":

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
