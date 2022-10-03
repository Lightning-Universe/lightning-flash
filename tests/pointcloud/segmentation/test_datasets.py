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
from unittest.mock import patch

import pytest

from flash.core.utilities.imports import _POINTCLOUD_TESTING
from flash.pointcloud.segmentation.datasets import LyftDataset, SemanticKITTIDataset


@pytest.mark.skipif(not _POINTCLOUD_TESTING, reason="pointcloud libraries aren't installed")
@patch("flash.pointcloud.segmentation.datasets.os.system")
def test_datasets(mock_system):
    LyftDataset("data")
    assert mock_system.call_count == 2
    assert "lyft" in mock_system.call_args_list[0][0][0]
    assert "data" in mock_system.call_args_list[0][0][0]
    assert "lyft" in mock_system.call_args_list[1][0][0]
    assert "data" in mock_system.call_args_list[1][0][0]

    mock_system.reset_mock()
    SemanticKITTIDataset("data")
    assert mock_system.call_count == 1
    assert "semantickitti" in mock_system.call_args_list[0][0][0]
    assert "data" in mock_system.call_args_list[0][0][0]
