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
import sys
from pathlib import Path
from unittest import mock

import pytest
import torch

from flash.core.utilities.imports import (
    _AUDIO_TESTING,
    _CORE_TESTING,
    _GRAPH_TESTING,
    _ICEVISION_AVAILABLE,
    _IMAGE_AVAILABLE,
    _IMAGE_EXTRAS_TESTING,
    _IMAGE_TESTING,
    _POINTCLOUD_TESTING,
    _TABULAR_TESTING,
    _TEXT_TESTING,
    _VIDEO_TESTING,
    _VISSL_AVAILABLE,
)
from tests.examples.utils import run_test
from tests.helpers.decorators import forked

root = Path(__file__).parent.parent.parent


@mock.patch.dict(os.environ, {"FLASH_TESTING": "1"})
@pytest.mark.parametrize(
    "file",
    [
        pytest.param(
            "audio_classification.py",
            marks=pytest.mark.skipif(not _AUDIO_TESTING, reason="audio libraries aren't installed"),
        ),
        pytest.param(
            "speech_recognition.py",
            marks=pytest.mark.skipif(not _AUDIO_TESTING, reason="audio libraries aren't installed"),
        ),
        pytest.param(
            "image_classification.py",
            marks=pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed"),
        ),
        pytest.param(
            "image_classification_multi_label.py",
            marks=pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed"),
        ),
        pytest.param(
            "image_embedder.py",
            marks=[
                pytest.mark.skipif(
                    not (_IMAGE_AVAILABLE and _VISSL_AVAILABLE), reason="image libraries aren't installed"
                ),
                pytest.mark.skipif(torch.cuda.device_count() > 1, reason="VISSL integration doesn't support multi-GPU"),
            ],
        ),
        pytest.param(
            "object_detection.py",
            marks=pytest.mark.skipif(
                not (_IMAGE_EXTRAS_TESTING and _ICEVISION_AVAILABLE), reason="image libraries aren't installed"
            ),
        ),
        pytest.param(
            "instance_segmentation.py",
            marks=pytest.mark.skipif(
                not (_IMAGE_EXTRAS_TESTING and _ICEVISION_AVAILABLE), reason="image libraries aren't installed"
            ),
        ),
        pytest.param(
            "keypoint_detection.py",
            marks=pytest.mark.skipif(
                not (_IMAGE_EXTRAS_TESTING and _ICEVISION_AVAILABLE), reason="image libraries aren't installed"
            ),
        ),
        pytest.param(
            "question_answering.py",
            marks=pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed"),
        ),
        pytest.param(
            "semantic_segmentation.py",
            marks=pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed"),
        ),
        pytest.param(
            "style_transfer.py",
            marks=[
                pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed"),
                pytest.mark.skipif(torch.cuda.device_count() >= 2, reason="PyStiche doesn't support DDP"),
            ],
        ),
        pytest.param(
            "summarization.py", marks=pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed")
        ),
        pytest.param(
            "tabular_classification.py",
            marks=pytest.mark.skipif(not _TABULAR_TESTING, reason="tabular libraries aren't installed"),
        ),
        pytest.param(
            "tabular_regression.py",
            marks=pytest.mark.skipif(not _TABULAR_TESTING, reason="tabular libraries aren't installed"),
        ),
        pytest.param(
            "tabular_forecasting.py",
            marks=pytest.mark.skipif(not _TABULAR_TESTING, reason="tabular libraries aren't installed"),
        ),
        pytest.param(
            "template.py",
            marks=[
                pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core."),
                pytest.mark.skipif(os.name == 'posix', reason="Flaky on Mac OS (CI)"),
                pytest.mark.skipif(sys.version_info >= (3, 9), reason="Undiagnosed segmentation fault in 3.9"),
            ],
        ),
        pytest.param(
            "text_classification.py",
            marks=pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed"),
        ),
        pytest.param(
            "text_embedder.py",
            marks=pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed"),
        ),
        # pytest.param(
        #     "text_classification_multi_label.py",
        #     marks=pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed")
        # ),
        pytest.param(
            "translation.py",
            marks=[
                pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed"),
                pytest.mark.skipif(os.name == "nt", reason="Encoding issues on Windows"),
            ],
        ),
        pytest.param(
            "video_classification.py",
            marks=pytest.mark.skipif(not _VIDEO_TESTING, reason="video libraries aren't installed"),
        ),
        pytest.param(
            "pointcloud_segmentation.py",
            marks=pytest.mark.skipif(not _POINTCLOUD_TESTING, reason="pointcloud libraries aren't installed"),
        ),
        pytest.param(
            "pointcloud_detection.py",
            marks=pytest.mark.skipif(not _POINTCLOUD_TESTING, reason="pointcloud libraries aren't installed"),
        ),
        pytest.param(
            "graph_classification.py",
            marks=pytest.mark.skipif(not _GRAPH_TESTING, reason="graph libraries aren't installed"),
        ),
        pytest.param(
            "graph_embedder.py",
            marks=pytest.mark.skipif(not _GRAPH_TESTING, reason="graph libraries aren't installed"),
        ),
    ],
)
@forked
def test_example(tmpdir, file):
    run_test(str(root / "flash_examples" / file))
