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
from typing import Any, Callable, Dict, List, Optional

from torch.utils.data import Dataset

from flash.core.data.data_module import DataModule
from flash.core.data.io.input import BaseDataFormat, DataKeys, Input, InputFormat
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.process import Deserializer
from flash.core.utilities.stages import RunningStage
from flash.pointcloud.detection.open3d_ml.inputs import (
    PointCloudObjectDetectionDataFormat,
    PointCloudObjectDetectorFoldersInput,
)


class PointCloudObjectDetectorDatasetInput(Input):
    def load_data(self, dataset: Dataset) -> Any:
        self.dataset = dataset
        return range(len(self.dataset))

    def load_sample(self, index: int) -> Any:
        sample = self.dataset[index]
        return {
            DataKeys.INPUT: sample["data"],
            DataKeys.METADATA: sample["attr"],
        }


class PointCloudObjectDetectorInputTransform(InputTransform):
    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        deserializer: Optional[Deserializer] = None,
        **_kwargs,
    ):

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            inputs={
                InputFormat.DATASETS: PointCloudObjectDetectorDatasetInput,
                InputFormat.FOLDERS: PointCloudObjectDetectorFoldersInput,
            },
            deserializer=deserializer,
            default_input=InputFormat.FOLDERS,
        )

    def get_state_dict(self):
        return {}

    def state_dict(self):
        return {}

    @classmethod
    def load_state_dict(cls, state_dict, strict: bool = False):
        pass


class PointCloudObjectDetectorData(DataModule):

    input_transform_cls = PointCloudObjectDetectorInputTransform

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[str] = None,
        val_folder: Optional[str] = None,
        test_folder: Optional[str] = None,
        predict_folder: Optional[str] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        scans_folder_name: Optional[str] = "scans",
        labels_folder_name: Optional[str] = "labels",
        calibrations_folder_name: Optional[str] = "calibs",
        data_format: Optional[BaseDataFormat] = PointCloudObjectDetectionDataFormat.KITTI,
        **data_module_kwargs: Any,
    ) -> "PointCloudObjectDetectorData":
        dataset_kwargs = dict(
            scans_folder_name=scans_folder_name,
            labels_folder_name=labels_folder_name,
            calibrations_folder_name=calibrations_folder_name,
            data_format=data_format,
        )
        return cls(
            PointCloudObjectDetectorFoldersInput(RunningStage.TRAINING, train_folder, **dataset_kwargs),
            PointCloudObjectDetectorFoldersInput(RunningStage.VALIDATING, val_folder, **dataset_kwargs),
            PointCloudObjectDetectorFoldersInput(RunningStage.TESTING, test_folder, **dataset_kwargs),
            PointCloudObjectDetectorFoldersInput(RunningStage.PREDICTING, predict_folder, **dataset_kwargs),
            input_transform=cls.input_transform_cls(
                train_transform,
                val_transform,
                test_transform,
                predict_transform,
            ),
            **data_module_kwargs,
        )

    @classmethod
    def from_files(
        cls,
        predict_files: Optional[List[str]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        scans_folder_name: Optional[str] = "scans",
        labels_folder_name: Optional[str] = "labels",
        calibrations_folder_name: Optional[str] = "calibs",
        data_format: Optional[BaseDataFormat] = PointCloudObjectDetectionDataFormat.KITTI,
        **data_module_kwargs: Any,
    ) -> "PointCloudObjectDetectorData":
        ds_kw = dict(
            scans_folder_name=scans_folder_name,
            labels_folder_name=labels_folder_name,
            calibrations_folder_name=calibrations_folder_name,
            data_format=data_format,
        )
        return cls(
            predict_dataset=PointCloudObjectDetectorFoldersInput(RunningStage.PREDICTING, predict_files, **ds_kw),
            input_transform=cls.input_transform_cls(predict_transform),
            **data_module_kwargs,
        )

    @classmethod
    def from_datasets(
        cls,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        predict_dataset: Optional[Dataset] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        **data_module_kwargs,
    ) -> "PointCloudObjectDetectorData":
        return cls(
            PointCloudObjectDetectorDatasetInput(RunningStage.TRAINING, train_dataset),
            PointCloudObjectDetectorDatasetInput(RunningStage.VALIDATING, val_dataset),
            PointCloudObjectDetectorDatasetInput(RunningStage.TESTING, test_dataset),
            PointCloudObjectDetectorDatasetInput(RunningStage.PREDICTING, predict_dataset),
            input_transform=cls.input_transform_cls(
                train_transform,
                val_transform,
                test_transform,
                predict_transform,
            ),
            **data_module_kwargs,
        )
