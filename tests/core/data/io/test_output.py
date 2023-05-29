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
from unittest.mock import Mock

import pytest
from flash.core.data.io.output import Output
from flash.core.utilities.imports import _TOPIC_CORE_AVAILABLE


@pytest.mark.skipif(not _TOPIC_CORE_AVAILABLE, reason="Not testing core.")
def test_output():
    """Tests basic ``Output`` methods."""
    my_output = Output()

    assert my_output.transform("test") == "test"

    my_output.transform = Mock()
    my_output("test")
    my_output.transform.assert_called_once()
