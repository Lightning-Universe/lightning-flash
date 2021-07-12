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
    _IMAGE_AVAILABLE,
    _PYSTICHE_GREATER_EQUAL_0_7_2,
    _PYTORCH_GEOMETRIC_AVAILABLE,
    _SKLEARN_AVAILABLE,
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
    "file",
    [
        pytest.param(
            "custom_task.py", marks=pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="sklearn isn't installed")
        ),
        pytest.param(
            "image_classification.py",
            marks=pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed")
        ),
        pytest.param(
            "image_classification_multi_label.py",
            marks=pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed")
        ),
        # pytest.param("finetuning", "object_detection.py"),  # TODO: takes too long.
        pytest.param(
            "semantic_segmentation.py",
            marks=pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed")
        ),
        pytest.param(
            "style_transfer.py",
            marks=pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed")
        ),
        pytest.param(
            "summarization.py", marks=pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed")
        ),
        pytest.param(
            "finetuning",
            "template.py",
            marks=pytest.mark.skipif(not _PYTORCH_GEOMETRIC_AVAILABLE, reason="pytorch geometric isn't installed")
        ),
        pytest.param(
            "predict",
            "image_classification.py",
            marks=pytest.mark.skipif(not _IMAGE_AVAILABLE, reason="image libraries aren't installed")
        ),
        pytest.param(
            "tabular_classification.py",
            marks=pytest.mark.skipif(not _TABULAR_TESTING, reason="tabular libraries aren't installed")
        ),
        pytest.param("template.py", marks=pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="sklearn isn't installed")),
        pytest.param(
            "text_classification.py",
            marks=pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed")
        ),
        # pytest.param(
        #     "text_classification_multi_label.py",
        #     marks=pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed")
        # ),
        pytest.param(
            "translation.py", marks=pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed")
        ),
        pytest.param(
            "video_classification.py",
            marks=pytest.mark.skipif(not _VIDEO_TESTING, reason="video libraries aren't installed")
        ),
        pytest.param(
            "predict",
            "template.py",
            marks=pytest.mark.skipif(not _PYTORCH_GEOMETRIC_AVAILABLE, reason="pytorch geometric isn't installed")
        ),
    ]
)
def test_example(tmpdir, file):
    run_test(str(Path(flash.PROJECT_ROOT) / "flash_examples" / file))
