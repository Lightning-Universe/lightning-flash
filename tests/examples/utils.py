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
from typing import List, Optional, Tuple


def call_script(
    filepath: str,
    args: Optional[List[str]] = None,
    timeout: Optional[int] = 60 * 10,
) -> Tuple[int, str, str]:
    with open(filepath) as original:
        data = original.readlines()

    with open(filepath, "w") as modified:
        modified.write("import pytorch_lightning as pl\npl.seed_everything(42)\n")
        modified.write("if __name__ == '__main__':\n")
        for line in data:
            modified.write(f"    {line}\n")

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

    with open(filepath, "w") as modified:
        modified.writelines(data)

    return p.returncode, stdout, stderr


def run_test(filepath):
    code, stdout, stderr = call_script(filepath)
    print(f"{filepath} STDOUT: {stdout}")
    print(f"{filepath} STDERR: {stderr}")
    assert not code, code
