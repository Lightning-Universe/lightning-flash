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
import pytest
import torch

from flash.core.utilities.imports import _POINTCLOUD_AVAILABLE, _POINTCLOUD_TESTING
from flash.pointcloud.segmentation import PointCloudSegmentation
from tests.helpers.task_tester import TaskTester


class TestPointCloudSegmentation(TaskTester):

    task = PointCloudSegmentation
    task_args = (2,)
    cli_command = "pointcloud_segmentation"
    is_testing = _POINTCLOUD_TESTING
    is_available = _POINTCLOUD_AVAILABLE


@pytest.mark.skipif(not _POINTCLOUD_TESTING, reason="pointcloud libraries aren't installed")
def test_backbones():
    backbones = PointCloudSegmentation.available_backbones()
    assert backbones == ["randlanet", "randlanet_s3dis", "randlanet_semantic_kitti", "randlanet_toronto3d"]


@pytest.mark.skipif(not _POINTCLOUD_TESTING, reason="pointcloud libraries aren't installed")
@pytest.mark.parametrize(
    "backbone",
    [
        "randlanet",
        "randlanet_s3dis",
        "randlanet_toronto3d",
        "randlanet_semantic_kitti",
    ],
)
def test_models(backbone):
    num_classes = 13
    model = PointCloudSegmentation(backbone=backbone, num_classes=num_classes)
    assert model.head.weight.shape == torch.Size([13, 32])
