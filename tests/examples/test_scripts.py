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

from flash.core.utilities.imports import _SKLEARN_AVAILABLE
from tests.examples.utils import run_test
from tests.helpers.utils import (
    _IMAGE_STLYE_TRANSFER_TESTING,
    _IMAGE_TESTING,
    _TABULAR_TESTING,
    _TEXT_TESTING,
    _VIDEO_TESTING,
)

root = Path(__file__).parent.parent.parent


@mock.patch.dict(os.environ, {"FLASH_TESTING": "1"})
@pytest.mark.parametrize(
    "folder, file",
    [
        pytest.param(
            "finetuning",
            "image_classification.py",
            marks=pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed")
        ),
        pytest.param(
            "finetuning",
            "image_classification_multi_label.py",
            marks=pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed")
        ),
        # pytest.param("finetuning", "object_detection.py"),  # TODO: takes too long.
        pytest.param(
            "finetuning",
            "semantic_segmentation.py",
            marks=pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed")
        ),
        pytest.param(
            "finetuning",
            "summarization.py",
            marks=pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed")
        ),
        pytest.param(
            "finetuning",
            "tabular_classification.py",
            marks=pytest.mark.skipif(not _TABULAR_TESTING, reason="tabular libraries aren't installed")
        ),
        # pytest.param("finetuning", "video_classification.py"),
        # pytest.param("finetuning", "text_classification.py"),  # TODO: takes too long
        pytest.param(
            "finetuning",
            "template.py",
            marks=pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="sklearn isn't installed")
        ),
        pytest.param(
            "finetuning",
            "translation.py",
            marks=pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed")
        ),
        pytest.param(
            "finetuning",
            "style_transfer.py",
            marks=pytest.mark.skipif(not _IMAGE_STLYE_TRANSFER_TESTING, reason="pystiche is not installed")
        ),
        pytest.param(
            "predict",
            "image_classification.py",
            marks=pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed")
        ),
        pytest.param(
            "predict",
            "image_classification_multi_label.py",
            marks=pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed")
        ),
        pytest.param(
            "predict",
            "semantic_segmentation.py",
            marks=pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed")
        ),
        pytest.param(
            "predict",
            "tabular_classification.py",
            marks=pytest.mark.skipif(not _TABULAR_TESTING, reason="tabular libraries aren't installed")
        ),
        # pytest.param("predict", "text_classification.py"),
        pytest.param(
            "predict",
            "image_embedder.py",
            marks=pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed")
        ),
        pytest.param(
            "predict",
            "video_classification.py",
            marks=pytest.mark.skipif(not _VIDEO_TESTING, reason="video libraries aren't installed")
        ),
        pytest.param(
            "predict",
            "summarization.py",
            marks=pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed")
        ),
        pytest.param(
            "predict",
            "template.py",
            marks=pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="sklearn isn't installed")
        ),
        pytest.param(
            "predict",
            "translation.py",
            marks=pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed")
        ),
    ]
)
def test_example(tmpdir, folder, file):
    run_test(str(root / "flash_examples" / folder / file))


@pytest.mark.skipif(reason="CI bug")
def test_generic_example(tmpdir):
    run_test(str(root / "flash_examples" / "generic_task.py"))


def test_custom_task(tmpdir):
    run_test(str(root / "flash_examples" / "custom_task.py"))
