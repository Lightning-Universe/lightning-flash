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
from unittest.mock import patch

import pytest

from flash.__main__ import main
from flash.core.utilities.imports import _TOPIC_AUDIO_AVAILABLE, _TOPIC_IMAGE_AVAILABLE


@pytest.mark.skipif(not _TOPIC_IMAGE_AVAILABLE, reason="image libraries aren't installed.")
@pytest.mark.skipif(not _TOPIC_AUDIO_AVAILABLE, reason="audio libraries aren't installed.")
def test_cli():
    cli_args = ["flash", "audio_classification", "--trainer.fast_dev_run", "True"]
    with patch("sys.argv", cli_args):
        try:
            main()
        except SystemExit:
            pass
