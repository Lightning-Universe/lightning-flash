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
import time
from collections import namedtuple

import pytest

from flash.core.data.utilities.classification import (
    get_target_details,
    get_target_formatter,
    get_target_mode,
    TargetMode,
)

Case = namedtuple("Case", ["target", "formatted_target", "target_mode", "labels", "num_classes"])

cases = [
    # Single
    Case([0, 1, 2], [0, 1, 2], TargetMode.SINGLE_NUMERIC, None, 3),
    Case([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 1, 2], TargetMode.SINGLE_BINARY, None, 3),
    Case(["blue", "green", "red"], [0, 1, 2], TargetMode.SINGLE_TOKEN, ["blue", "green", "red"], 3),
    # Multi
    Case([[0, 1], [1, 2], [2, 0]], [[1, 1, 0], [0, 1, 1], [1, 0, 1]], TargetMode.MULTI_NUMERIC, None, 3),
    Case([[1, 1, 0], [0, 1, 1], [1, 0, 1]], [[1, 1, 0], [0, 1, 1], [1, 0, 1]], TargetMode.MULTI_BINARY, None, 3),
    Case(
        [["blue", "green"], ["green", "red"], ["red", "blue"]],
        [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
        TargetMode.MULTI_TOKEN,
        ["blue", "green", "red"],
        3,
    ),
    Case(
        ["blue,green", "green,red", "red,blue"],
        [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
        TargetMode.MUTLI_COMMA_DELIMITED,
        ["blue", "green", "red"],
        3,
    ),
    # Ambiguous
    Case([[0], [1, 2], [2, 0]], [[1, 0, 0], [0, 1, 1], [1, 0, 1]], TargetMode.MULTI_NUMERIC, None, 3),
    Case([[1, 0, 0], [0, 1, 1], [1, 0, 1]], [[1, 0, 0], [0, 1, 1], [1, 0, 1]], TargetMode.MULTI_BINARY, None, 3),
    Case(
        [["blue"], ["green", "red"], ["red", "blue"]],
        [[1, 0, 0], [0, 1, 1], [1, 0, 1]],
        TargetMode.MULTI_TOKEN,
        ["blue", "green", "red"],
        3,
    ),
    Case(
        ["blue", "green,red", "red,blue"],
        [[1, 0, 0], [0, 1, 1], [1, 0, 1]],
        TargetMode.MUTLI_COMMA_DELIMITED,
        ["blue", "green", "red"],
        3,
    ),
    # Special cases
    Case(["blue ", " green", "red"], [0, 1, 2], TargetMode.SINGLE_TOKEN, ["blue", "green", "red"], 3),
    Case(
        ["blue", "green, red", "red, blue"],
        [[1, 0, 0], [0, 1, 1], [1, 0, 1]],
        TargetMode.MUTLI_COMMA_DELIMITED,
        ["blue", "green", "red"],
        3,
    ),
]


@pytest.mark.parametrize("case", cases)
def test_case(case):
    target_mode = get_target_mode(case.target)
    assert target_mode is case.target_mode

    labels, num_classes = get_target_details(case.target, target_mode)
    assert labels == case.labels
    assert num_classes == case.num_classes

    formatter = get_target_formatter(target_mode, labels, num_classes)
    assert [formatter(t) for t in case.target] == case.formatted_target


@pytest.mark.parametrize("case", cases)
def test_speed(case):
    targets = case.target * 100000  # 300000 targets

    start = time.time()
    target_mode = get_target_mode(targets)
    labels, num_classes = get_target_details(targets, target_mode)
    formatter = get_target_formatter(target_mode, labels, num_classes)
    end = time.time()

    assert (end - start) / 300000 < 1e-5  # 0.01ms per target

    start = time.time()
    _ = [formatter(t) for t in targets]
    end = time.time()

    assert (end - start) / 300000 < 1e-5  # 0.01ms per target
