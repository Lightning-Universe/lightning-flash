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
from pathlib import Path
from unittest import mock

import pytest

from flash.core.utilities.imports import (
    _AUDIO_TESTING,
    _GRAPH_TESTING,
    _ICEVISION_AVAILABLE,
    _IMAGE_AVAILABLE,
    _IMAGE_TESTING,
    _POINTCLOUD_TESTING,
    _SKLEARN_AVAILABLE,
    _TABULAR_TESTING,
    _TEXT_TESTING,
    _VIDEO_TESTING,
)
from tests.examples.utils import run_test

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
            "object_detection.py",
            marks=pytest.mark.skipif(
                not (_IMAGE_AVAILABLE and _ICEVISION_AVAILABLE), reason="image libraries aren't installed"
            ),
        ),
        pytest.param(
            "instance_segmentation.py",
            marks=pytest.mark.skipif(
                not (_IMAGE_AVAILABLE and _ICEVISION_AVAILABLE), reason="image libraries aren't installed"
            ),
        ),
        pytest.param(
            "keypoint_detection.py",
            marks=pytest.mark.skipif(
                not (_IMAGE_AVAILABLE and _ICEVISION_AVAILABLE), reason="image libraries aren't installed"
            ),
        ),
        # pytest.param("finetuning", "object_detection.py"),  # TODO: takes too long.
        pytest.param(
            "question_answering.py",
            marks=pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed"),
        ),
        pytest.param(
            "semantic_segmentation.py",
            marks=pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed"),
        ),
        pytest.param(
            "style_transfer.py", marks=pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed")
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
        pytest.param("template.py", marks=pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="sklearn isn't installed")),
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
@pytest.mark.forked
def test_example(tmpdir, file):
    run_test(str(root / "flash_examples" / file))
