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

from flash.core.utilities.imports import _BAAL_AVAILABLE, _FIFTYONE_AVAILABLE, _IMAGE_AVAILABLE, _LEARN2LEARN_AVAILABLE
from tests.examples.utils import run_test

root = Path(__file__).parent.parent.parent


@mock.patch.dict(os.environ, {"FLASH_TESTING": "1"})
@pytest.mark.parametrize(
    "folder, file",
    [
        pytest.param(
            "fiftyone",
            "image_classification.py",
            marks=pytest.mark.skipif(
                not (_IMAGE_AVAILABLE and _FIFTYONE_AVAILABLE), reason="fiftyone library isn't installed"
            ),
        ),
        pytest.param(
            "baal",
            "image_classification_active_learning.py",
            marks=pytest.mark.skipif(not (_IMAGE_AVAILABLE and _BAAL_AVAILABLE), reason="baal library isn't installed"),
        ),
        pytest.param(
            "learn2learn",
            "image_classification_imagenette_mini.py",
            marks=pytest.mark.skipif(
                not (_IMAGE_AVAILABLE and _LEARN2LEARN_AVAILABLE), reason="learn2learn isn't installed"
            ),
        ),
    ],
)
def test_integrations(tmpdir, folder, file):
    run_test(str(root / "flash_examples" / "integrations" / folder / file))
