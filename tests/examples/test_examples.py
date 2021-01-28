import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

root = Path(__file__).parent.parent.parent


def call_script(filepath: str,
                args: Optional[List[str]] = None,
                timeout: Optional[int] = 60 * 5) -> Tuple[int, str, str]:
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
    assert not code
    print(f"{filepath} STDOUT: {stdout}")
    print(f"{filepath} STDERR: {stderr}")


@pytest.mark.parametrize(
    "step,file",
    [
        ("finetuning", "image_classification.py"),
        ("finetuning", "tabular_classification.py"),
        ("predict", "classify_image.py"),
        ("predict", "classify_tabular.py"),
        # "classify_text.py" TODO: takes too long
    ]
)
def test_finetune_example(tmpdir, step, file):
    with tmpdir.as_cwd():
        run_test(str(root / "flash_examples" / step / file))


def test_generic_example(tmpdir):
    with tmpdir.as_cwd():
        run_test(str(root / "flash_examples" / "generic_task.py"))
