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

from flash.core.utilities.imports import (
    _AUDIO_AVAILABLE,
    _GRAPH_AVAILABLE,
    _IMAGE_AVAILABLE,
    _POINTCLOUD_AVAILABLE,
    _SERVE_AVAILABLE,
    _TABULAR_AVAILABLE,
    _TEXT_AVAILABLE,
    _VIDEO_AVAILABLE,
)

_IMAGE_TESTING = _IMAGE_AVAILABLE
_VIDEO_TESTING = _VIDEO_AVAILABLE
_TABULAR_TESTING = _TABULAR_AVAILABLE
_TEXT_TESTING = _TEXT_AVAILABLE
_SERVE_TESTING = _SERVE_AVAILABLE
_POINTCLOUD_TESTING = _POINTCLOUD_AVAILABLE
_GRAPH_TESTING = _GRAPH_AVAILABLE
_AUDIO_TESTING = _AUDIO_AVAILABLE

if "FLASH_TEST_TOPIC" in os.environ:
    topic = os.environ["FLASH_TEST_TOPIC"]
    _IMAGE_TESTING = topic == "image"
    _VIDEO_TESTING = topic == "video"
    _TABULAR_TESTING = topic == "tabular"
    _TEXT_TESTING = topic == "text"
    _SERVE_TESTING = topic == "serve"
    _POINTCLOUD_TESTING = topic == "pointcloud"
    _GRAPH_TESTING = topic == "graph"
    _AUDIO_TESTING = topic == "audio"
