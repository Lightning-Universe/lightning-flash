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
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from unittest import mock

import pytest

from flash.core.utilities.imports import (
    _IMAGE_AVAILABLE,
    _PYSTICHE_GREATER_EQUAL_0_7_2,
    _TABULAR_AVAILABLE,
    _TEXT_AVAILABLE,
    _TORCHVISION_GREATER_EQUAL_0_9,
    _VIDEO_AVAILABLE,
)

_IMAGE_AVAILABLE = _IMAGE_AVAILABLE and _TORCHVISION_GREATER_EQUAL_0_9

root = Path(__file__).parent.parent.parent


def call_script(
    filepath: str,
    args: Optional[List[str]] = None,
    timeout: Optional[int] = 60 * 5,
) -> Tuple[int, str, str]:
    if args is None:
        args = []
    args = [str(a) for a in args]
    command = [sys.executable, "-m", "coverage", "run", filepath] + args
    print(" ".join(command))
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        stdout, stderr = p.communicate()
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")
    return p.returncode, stdout, stderr


def run_test(filepath):
    code, stdout, stderr = call_script(filepath)
    print(f"{filepath} STDOUT: {stdout}")
    print(f"{filepath} STDERR: {stderr}")
    assert not code


@mock.patch.dict(os.environ, {"FLASH_TESTING": "1"})
@pytest.mark.parametrize(
    "folder, file",
    [
        pytest.param(
            "finetuning",
            "image_classification.py",
            marks=pytest.mark.skipif(not _IMAGE_AVAILABLE, reason="image libraries aren't installed")
        ),
        pytest.param(
            "finetuning",
            "image_classification_multi_label.py",
            marks=pytest.mark.skipif(not _IMAGE_AVAILABLE, reason="image libraries aren't installed")
        ),
        # pytest.param("finetuning", "object_detection.py"),  # TODO: takes too long.
        pytest.param(
            "finetuning",
            "semantic_segmentation.py",
            marks=pytest.mark.skipif(not _IMAGE_AVAILABLE, reason="image libraries aren't installed")
        ),
        # pytest.param("finetuning", "summarization.py"),  # TODO: takes too long.
        pytest.param(
            "finetuning",
            "tabular_classification.py",
            marks=pytest.mark.skipif(not _TABULAR_AVAILABLE, reason="tabular libraries aren't installed")
        ),
        # pytest.param("finetuning", "video_classification.py"),
        # pytest.param("finetuning", "text_classification.py"),  # TODO: takes too long
        pytest.param(
            "finetuning",
            "translation.py",
            marks=pytest.mark.skipif(not _TEXT_AVAILABLE, reason="text libraries aren't installed")
        ),
        pytest.param(
            "finetuning",
            "style_transfer.py",
            marks=pytest.mark.skipif(not _PYSTICHE_GREATER_EQUAL_0_7_2, reason="pystiche is not installed")
        ),
        pytest.param(
            "predict",
            "image_classification.py",
            marks=pytest.mark.skipif(not _IMAGE_AVAILABLE, reason="image libraries aren't installed")
        ),
        pytest.param(
            "predict",
            "image_classification_multi_label.py",
            marks=pytest.mark.skipif(not _IMAGE_AVAILABLE, reason="image libraries aren't installed")
        ),
        pytest.param(
            "predict",
            "semantic_segmentation.py",
            marks=pytest.mark.skipif(not _IMAGE_AVAILABLE, reason="image libraries aren't installed")
        ),
        pytest.param(
            "predict",
            "tabular_classification.py",
            marks=pytest.mark.skipif(not _TABULAR_AVAILABLE, reason="tabular libraries aren't installed")
        ),
        # pytest.param("predict", "text_classification.py"),
        pytest.param(
            "predict",
            "image_embedder.py",
            marks=pytest.mark.skipif(not _IMAGE_AVAILABLE, reason="image libraries aren't installed")
        ),
        pytest.param(
            "predict",
            "video_classification.py",
            marks=pytest.mark.skipif(not _VIDEO_AVAILABLE, reason="video libraries aren't installed")
        ),
        # pytest.param("predict", "summarization.py"),  # TODO: takes too long
        pytest.param(
            "predict",
            "translation.py",
            marks=pytest.mark.skipif(not _TEXT_AVAILABLE, reason="text libraries aren't installed")
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
