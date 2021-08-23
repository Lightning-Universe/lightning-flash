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

from flash.pointcloud.detection import PointCloudObjectDetector
from tests.helpers.utils import _POINTCLOUD_TESTING


@pytest.mark.skipif(not _POINTCLOUD_TESTING, reason="pointcloud libraries aren't installed")
def test_backbones():

    backbones = PointCloudObjectDetector.available_backbones()
    assert backbones == ["pointpillars", "pointpillars_kitti"]
