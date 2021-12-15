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

import numpy as np
import pytest
import torch

from flash.core.data.utilities.classification import (
    get_target_details,
    get_target_formatter,
    get_target_mode,
    TargetMode,
)
from tests.helpers.retry import retry

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
    Case(
        ["blue green", "green red", "red blue"],
        [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
        TargetMode.MUTLI_SPACE_DELIMITED,
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
    Case(
        ["blue", "green red", "red blue"],
        [[1, 0, 0], [0, 1, 1], [1, 0, 1]],
        TargetMode.MUTLI_SPACE_DELIMITED,
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
    Case(
        ["blue", "green ,red", "red ,blue"],
        [[1, 0, 0], [0, 1, 1], [1, 0, 1]],
        TargetMode.MUTLI_COMMA_DELIMITED,
        ["blue", "green", "red"],
        3,
    ),
    Case(
        [f"class_{i}" for i in range(10000)],
        list(range(10000)),
        TargetMode.SINGLE_TOKEN,
        [f"class_{i}" for i in range(10000)],
        10000,
    ),
    # Array types
    Case(torch.tensor([[0], [1]]), [0, 1], TargetMode.SINGLE_NUMERIC, None, 2),
    Case(torch.tensor([0, 1, 2]), [0, 1, 2], TargetMode.SINGLE_NUMERIC, None, 3),
    Case(np.array([0, 1, 2]), [0, 1, 2], TargetMode.SINGLE_NUMERIC, None, 3),
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
@retry(3)
def test_speed(case):
    repeats = int(1e5 / len(case.target))  # Approx. a hundred thousand targets

    if torch.is_tensor(case.target):
        targets = case.target.repeat(repeats, *(1 for _ in range(case.target.ndim - 1)))
    elif isinstance(case.target, np.ndarray):
        targets = np.repeat(case.target, repeats)
    else:
        targets = case.target * repeats

    start = time.perf_counter()
    target_mode = get_target_mode(targets)
    labels, num_classes = get_target_details(targets, target_mode)
    formatter = get_target_formatter(target_mode, labels, num_classes)
    end = time.perf_counter()

    assert (end - start) / len(targets) < 1e-5  # 0.01ms per target

    start = time.perf_counter()
    _ = [formatter(t) for t in targets]
    end = time.perf_counter()

    assert (end - start) / len(targets) < 1e-5  # 0.01ms per target
