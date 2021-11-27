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

from flash.core.data.utilities.classification import (
    get_target_details,
    get_target_formatter,
    get_target_mode,
    TargetMode,
)


@pytest.mark.parametrize(
    "targets, expected_mode",
    [
        # Test single
        ([0, 1, 2], TargetMode.SINGLE_NUMERIC),
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], TargetMode.SINGLE_BINARY),
        (["red", "green", "blue"], TargetMode.SINGLE_TOKEN),
        # Test multi
        ([[0, 1], [1, 2], [2, 0]], TargetMode.MULTI_NUMERIC),
        ([[1, 1, 0], [0, 1, 1], [1, 0, 1]], TargetMode.MULTI_BINARY),
        ([["red", "green"], ["green", "blue"], ["blue", "red"]], TargetMode.MULTI_TOKEN),
        (["red,green", "green,blue", "blue,red"], TargetMode.MUTLI_COMMA_DELIMITED),
        # Test ambiguous targets
        ([[0], [1], [2]], TargetMode.SINGLE_NUMERIC),
        ([[0], [1, 2], [2, 0]], TargetMode.MULTI_NUMERIC),
        ([[1, 0, 0], [0, 1, 1], [1, 0, 1]], TargetMode.MULTI_BINARY),
        ([["red"], ["green", "blue"], ["blue", "red"]], TargetMode.MULTI_TOKEN),
        (["red", "green,blue", "blue,red"], TargetMode.MUTLI_COMMA_DELIMITED),
        ([["red"], ["green"], ["blue"]], TargetMode.SINGLE_TOKEN),
    ],
)
def test_get_target_mode(targets, expected_mode):
    assert get_target_mode(targets) is expected_mode


@pytest.mark.parametrize(
    "targets, formatted_targets",
    [
        # Test single
        ([0, 1, 2], [0, 1, 2]),
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 1, 2]),
        (["red", "green", "blue"], [2, 1, 0]),
        # Test multi
        ([[0, 1], [1, 2], [2, 0]], [[1, 1, 0], [0, 1, 1], [1, 0, 1]]),
        ([[1, 1, 0], [0, 1, 1], [1, 0, 1]], [[1, 1, 0], [0, 1, 1], [1, 0, 1]]),
        ([["red", "green"], ["green", "blue"], ["blue", "red"]], [[0, 1, 1], [1, 1, 0], [1, 0, 1]]),
        (["red,green", "green,blue", "blue,red"], [[0, 1, 1], [1, 1, 0], [1, 0, 1]]),
        # Test ambiguous targets
        ([[0], [1], [2]], [0, 1, 2]),
        ([[0], [1, 2], [2, 0]], [[1, 0, 0], [0, 1, 1], [1, 0, 1]]),
        ([[1, 0, 0], [0, 1, 1], [1, 0, 1]], [[1, 0, 0], [0, 1, 1], [1, 0, 1]]),
        ([["red"], ["green", "blue"], ["blue", "red"]], [[0, 0, 1], [1, 1, 0], [1, 0, 1]]),
        (["red", "green,blue", "blue,red"], [[0, 0, 1], [1, 1, 0], [1, 0, 1]]),
        ([["red"], ["green"], ["blue"]], [2, 1, 0]),
    ],
)
def test_target_format(targets, formatted_targets):
    target_mode = get_target_mode(targets)
    labels, num_classes = get_target_details(targets, target_mode)
    formatter = get_target_formatter(target_mode, labels, num_classes)
    assert [formatter(target) for target in targets] == formatted_targets
