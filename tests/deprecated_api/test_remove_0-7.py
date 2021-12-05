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
import pytest

from flash.core.classification import Labels, Probabilities


def test_v0_7_deprecated_labels(tmpdir):
    with pytest.deprecated_call(match="`Labels` was deprecated in v0.6.0 and will be removed in v0.7.0."):
        Labels()


def test_v0_7_deprecated_probabilities(tmpdir):
    with pytest.deprecated_call(match="`Probabilities` was deprecated in v0.6.0 and will be removed in v0.7.0."):
        Probabilities()
