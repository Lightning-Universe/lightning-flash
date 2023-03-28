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
    CommaDelimitedMultiLabelTargetFormatter,
    MultiBinaryTargetFormatter,
    MultiLabelTargetFormatter,
    MultiNumericTargetFormatter,
    MultiSoftTargetFormatter,
    SingleBinaryTargetFormatter,
    SingleLabelTargetFormatter,
    SingleNumericTargetFormatter,
    SpaceDelimitedTargetFormatter,
    get_target_formatter,
)
from flash.core.utilities.imports import _TOPIC_CORE_AVAILABLE

Case = namedtuple("Case", ["target", "formatted_target", "target_formatter_type", "labels", "num_classes"])

cases = [
    # Single
    Case([0, 1, 2], [0, 1, 2], SingleNumericTargetFormatter, None, 3),
    Case([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 1, 2], SingleBinaryTargetFormatter, None, 3),
    Case([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 1, 2], SingleBinaryTargetFormatter, None, 3),
    Case(["blue", "green", "red"], [0, 1, 2], SingleLabelTargetFormatter, ["blue", "green", "red"], 3),
    # Multi
    Case([[0, 1], [1, 2], [2, 0]], [[1, 1, 0], [0, 1, 1], [1, 0, 1]], MultiNumericTargetFormatter, None, 3),
    Case([[1, 1, 0], [0, 1, 1], [1, 0, 1]], [[1, 1, 0], [0, 1, 1], [1, 0, 1]], MultiBinaryTargetFormatter, None, 3),
    Case(
        [[0.1, 0.9, 0], [0, 0.7, 0.6], [0.5, 0, 0.4]],
        [[0.1, 0.9, 0], [0, 0.7, 0.6], [0.5, 0, 0.4]],
        MultiSoftTargetFormatter,
        None,
        3,
    ),
    Case(
        [["blue", "green"], ["green", "red"], ["red", "blue"]],
        [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
        MultiLabelTargetFormatter,
        ["blue", "green", "red"],
        3,
    ),
    Case(
        ["blue,green", "green,red", "red,blue"],
        [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
        CommaDelimitedMultiLabelTargetFormatter,
        ["blue", "green", "red"],
        3,
    ),
    Case(
        ["blue green", "green red", "red blue"],
        [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
        SpaceDelimitedTargetFormatter,
        ["blue", "green", "red"],
        3,
    ),
    # Ambiguous
    Case([[0], [0, 1], [1, 2]], [[1, 0, 0], [1, 1, 0], [0, 1, 1]], MultiNumericTargetFormatter, None, 3),
    Case([[1, 0, 0], [0, 1, 1], [1, 0, 1]], [[1, 0, 0], [0, 1, 1], [1, 0, 1]], MultiBinaryTargetFormatter, None, 3),
    Case(
        [[1, 0, 0], [0, 0.9, 0.7], [0.6, 0, 0.5]],
        [[1, 0, 0], [0, 0.9, 0.7], [0.6, 0, 0.5]],
        MultiSoftTargetFormatter,
        None,
        3,
    ),
    Case(
        [[1, 0, 1], [0, 0.9, 0.7], [0.6, 0, 0.5]],
        [[1, 0, 1], [0, 0.9, 0.7], [0.6, 0, 0.5]],
        MultiSoftTargetFormatter,
        None,
        3,
    ),
    Case(
        [["blue"], ["green", "red"], ["red", "blue"]],
        [[1, 0, 0], [0, 1, 1], [1, 0, 1]],
        MultiLabelTargetFormatter,
        ["blue", "green", "red"],
        3,
    ),
    Case(
        ["blue", "green,red", "red,blue"],
        [[1, 0, 0], [0, 1, 1], [1, 0, 1]],
        CommaDelimitedMultiLabelTargetFormatter,
        ["blue", "green", "red"],
        3,
    ),
    Case(
        ["blue", "green red", "red blue"],
        [[1, 0, 0], [0, 1, 1], [1, 0, 1]],
        SpaceDelimitedTargetFormatter,
        ["blue", "green", "red"],
        3,
    ),
    # Special cases
    Case(["blue ", " green", "red"], [0, 1, 2], SingleLabelTargetFormatter, ["blue", "green", "red"], 3),
    Case(
        ["blue", "green, red", "red, blue"],
        [[1, 0, 0], [0, 1, 1], [1, 0, 1]],
        CommaDelimitedMultiLabelTargetFormatter,
        ["blue", "green", "red"],
        3,
    ),
    Case(
        ["blue", "green ,red", "red ,blue"],
        [[1, 0, 0], [0, 1, 1], [1, 0, 1]],
        CommaDelimitedMultiLabelTargetFormatter,
        ["blue", "green", "red"],
        3,
    ),
    Case(
        [f"class_{i}" for i in range(10000)],
        list(range(10000)),
        SingleLabelTargetFormatter,
        [f"class_{i}" for i in range(10000)],
        10000,
    ),
    # Array types
    Case(torch.tensor([[0], [1]]), [0, 1], SingleNumericTargetFormatter, None, 2),
    Case(torch.tensor([0, 1, 2]), [0, 1, 2], SingleNumericTargetFormatter, None, 3),
    Case(np.array([0, 1, 2]), [0, 1, 2], SingleNumericTargetFormatter, None, 3),
]


@pytest.mark.skipif(not _TOPIC_CORE_AVAILABLE, reason="Not testing core.")
@pytest.mark.parametrize("case", cases)
def test_case(case):
    formatter = get_target_formatter(case.target)

    assert isinstance(formatter, case.target_formatter_type)
    assert formatter.labels == case.labels
    assert formatter.num_classes == case.num_classes
    assert [formatter(t) for t in case.target] == case.formatted_target


@pytest.mark.skipif(not _TOPIC_CORE_AVAILABLE, reason="Not testing core.")
@pytest.mark.parametrize("case", cases)
def test_speed(case):
    repeats = int(1e5 / len(case.target))  # Approx. a hundred thousand targets

    if torch.is_tensor(case.target):
        targets = case.target.repeat(repeats, *(1 for _ in range(case.target.ndim - 1)))
    elif isinstance(case.target, np.ndarray):
        targets = np.repeat(case.target, repeats)
    else:
        targets = case.target * repeats

    start = time.perf_counter()
    formatter = get_target_formatter(targets)
    end = time.perf_counter()

    assert (end - start) / len(targets) < 1e-4  # 0.1ms per target

    start = time.perf_counter()
    _ = [formatter(t) for t in targets]
    end = time.perf_counter()

    assert (end - start) / len(targets) < 1e-4  # 0.1ms per target
