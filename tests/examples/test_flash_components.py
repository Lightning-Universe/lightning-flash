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

from tests.examples.utils import run_test
from tests.helpers.utils import _IMAGE_TESTING

root = Path(__file__).parent.parent.parent


@mock.patch.dict(os.environ, {"FLASH_TESTING": "1"})
@pytest.mark.parametrize(
    "folder, file",
    [
        pytest.param(
            "flash_components",
            "custom_data_loading.py",
            marks=pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed"),
        ),
    ],
)
def test_components(folder, file):
    run_test(str(Path(__file__) / "../../flash_examples" / folder / file))
