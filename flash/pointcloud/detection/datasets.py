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
import os

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _POINTCLOUD_AVAILABLE
from flash.pointcloud.segmentation.datasets import executor

if _POINTCLOUD_AVAILABLE:
    from open3d.ml.datasets import KITTI

_OBJECT_DETECTION_DATASET = FlashRegistry("dataset")


@_OBJECT_DETECTION_DATASET
def kitti(dataset_path, download, **kwargs):
    name = "KITTI"
    download_path = os.path.join(dataset_path, name, "Kitti")
    if not os.path.exists(download_path):
        executor(
            "https://raw.githubusercontent.com/intel-isl/Open3D-ML/master/scripts/download_datasets/download_kitti.sh",  # noqa E501
            None,
            dataset_path,
            name,
        )
    return KITTI(download_path, **kwargs)


def KITTIDataset(dataset_path, download: bool = True, **kwargs):
    return _OBJECT_DETECTION_DATASET.get("kitti")(dataset_path, download, **kwargs)
