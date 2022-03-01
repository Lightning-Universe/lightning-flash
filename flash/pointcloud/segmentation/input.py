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
from typing import Any

from torch.utils.data import Dataset

from flash.core.data.io.input import DataKeys, Input
from flash.core.utilities.imports import requires
from flash.pointcloud.segmentation.open3d_ml.sequences_dataset import SequencesDataset


class PointCloudSegmentationDatasetInput(Input):
    num_classes: Dataset
    dataset: Dataset

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
