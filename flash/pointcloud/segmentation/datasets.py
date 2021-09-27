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

if _POINTCLOUD_AVAILABLE:
    from open3d.ml.datasets import Lyft, SemanticKITTI

_SEGMENTATION_DATASET = FlashRegistry("dataset")


def executor(download_script, preprocess_script, dataset_path, name):
    if not os.path.exists(os.path.join(dataset_path, name)):
        os.system(f'bash -c "bash <(curl -s {download_script}) {dataset_path}"')
        if preprocess_script:
            os.system(f'bash -c "bash <(curl -s {preprocess_script}) {dataset_path}"')


@_SEGMENTATION_DATASET
def lyft(dataset_path):
    name = "Lyft"
    executor(
        "https://raw.githubusercontent.com/intel-isl/Open3D-ML/master/scripts/download_datasets/download_lyft.sh",
        "https://github.com/intel-isl/Open3D-ML/blob/master/scripts/preprocess_lyft.py",
        dataset_path,
        name,
    )
    return Lyft(os.path.join(dataset_path, name))


def LyftDataset(dataset_path):
    return _SEGMENTATION_DATASET.get("lyft")(dataset_path)


@_SEGMENTATION_DATASET
def semantickitti(dataset_path, download, **kwargs):
    name = "SemanticKitti"
    if download:
        executor(
            "https://raw.githubusercontent.com/intel-isl/Open3D-ML/master/scripts/download_datasets/download_semantickitti.sh",  # noqa E501
            None,
            dataset_path,
            name,
        )
    return SemanticKITTI(os.path.join(dataset_path, name), **kwargs)


def SemanticKITTIDataset(dataset_path, download: bool = True, **kwargs):
    return _SEGMENTATION_DATASET.get("semantickitti")(dataset_path, download, **kwargs)
