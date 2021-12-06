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
from typing import Any, Callable, Dict, List, Optional, Tuple

from torch.utils.data import Dataset

from flash.core.data.data_module import DataModule
from flash.core.data.io.input import DataKeys, InputFormat
from flash.core.data.io.input_base import Input
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.process import Deserializer
from flash.core.utilities.imports import requires
from flash.core.utilities.stages import RunningStage
from flash.pointcloud.segmentation.open3d_ml.sequences_dataset import SequencesDataset


class PointCloudSegmentationDatasetInput(Input):
    @requires("pointcloud")
    def load_data(self, dataset: Dataset) -> Any:
        if self.training and hasattr(dataset, "num_classes"):
            self.num_classes = dataset.num_classes
        self.dataset = dataset
        return range(len(self.dataset))

    def load_sample(self, index: int) -> Any:
        sample = self.dataset[index]
        return {
            DataKeys.INPUT: sample["data"],
            DataKeys.METADATA: sample["attr"],
        }


class PointCloudSegmentationFoldersInput(PointCloudSegmentationDatasetInput):
    @requires("pointcloud")
    def load_data(self, folder: str) -> Any:
        return super().load_data(SequencesDataset(folder, use_cache=True, predicting=self.predicting))


class PointCloudSegmentationInputTransform(InputTransform):
    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        image_size: Tuple[int, int] = (196, 196),
        deserializer: Optional[Deserializer] = None,
    ):
        self.image_size = image_size

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            inputs={
                InputFormat.DATASETS: PointCloudSegmentationDatasetInput,
                InputFormat.FOLDERS: PointCloudSegmentationFoldersInput,
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


class PointCloudSegmentationData(DataModule):

    input_transform_cls = PointCloudSegmentationInputTransform

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
        **data_module_kwargs: Any,
    ) -> "PointCloudSegmentationData":
        return cls(
            PointCloudSegmentationFoldersInput(RunningStage.TRAINING, train_folder),
            PointCloudSegmentationFoldersInput(RunningStage.VALIDATING, val_folder),
            PointCloudSegmentationFoldersInput(RunningStage.TESTING, test_folder),
            PointCloudSegmentationFoldersInput(RunningStage.PREDICTING, predict_folder),
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
        **data_module_kwargs: Any,
    ) -> "PointCloudSegmentationData":
        return cls(
            predict_dataset=PointCloudSegmentationFoldersInput(RunningStage.PREDICTING, predict_files),
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
        **data_module_kwargs: Any,
    ) -> "PointCloudSegmentationData":
        return cls(
            PointCloudSegmentationDatasetInput(RunningStage.TRAINING, train_dataset),
            PointCloudSegmentationDatasetInput(RunningStage.VALIDATING, val_dataset),
            PointCloudSegmentationDatasetInput(RunningStage.TESTING, test_dataset),
            PointCloudSegmentationDatasetInput(RunningStage.PREDICTING, predict_dataset),
            input_transform=cls.input_transform_cls(
                train_transform,
                val_transform,
                test_transform,
                predict_transform,
            ),
            **data_module_kwargs,
        )
