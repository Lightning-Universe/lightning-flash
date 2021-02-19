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
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

root = Path(__file__).parent.parent.parent


def call_script(
    filepath: str,
    args: Optional[List[str]] = None,
    timeout: Optional[int] = 60 * 5,
) -> Tuple[int, str, str]:
    if args is None:
        args = []
    args = [str(a) for a in args]
    command = [sys.executable, filepath] + args
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


@pytest.mark.parametrize(
    "step,file",
    [
        ("finetuning", "image_classification.py"),
        # ("finetuning", "object_detection.py"),  # TODO: takes too long.
        # ("finetuning", "summarization.py"),  # TODO: takes too long.
        ("finetuning", "tabular_classification.py"),
        ("finetuning", "text_classification.py"),
        # ("finetuning", "translation.py"),  # TODO: takes too long.
        ("predict", "classify_image.py"),
        ("predict", "classify_tabular.py"),
        ("predict", "classify_text.py"),
        ("predict", "image_embedder.py"),
        ("predict", "summarize.py"),
        # ("predict", "translate.py"),  # TODO: takes too long
    ]
)
def test_example(tmpdir, step, file):
    run_test(str(root / "flash_examples" / step / file))


def test_generic_example(tmpdir):
    run_test(str(root / "flash_examples" / "generic_task.py"))
