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
from collections import namedtuple

import pytest
import torch

from flash.core.data.batch import default_uncollate
from flash.core.utilities.imports import _CORE_TESTING

Case = namedtuple("Case", ["collated_batch", "uncollated_batch"])

cases = [
    # Primitives
    Case({"preds": [1, 2, 3]}, [{"preds": 1}, {"preds": 2}, {"preds": 3}]),
    Case(
        {"preds": [1, 2, 3], "metadata": [4, 5, 6]},
        [{"preds": 1, "metadata": 4}, {"preds": 2, "metadata": 5}, {"preds": 3, "metadata": 6}],
    ),
    Case(([1, 2, 3], [4, 5, 6]), [[1, 2, 3], [4, 5, 6]]),
    Case([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]),
    Case([[1, 2], [4, 5, 6]], [[1, 2], [4, 5, 6]]),
    Case([["a", "b"], ["a", "c", "d"]], [["a", "b"], ["a", "c", "d"]]),
    # Tensors
    Case({"preds": torch.tensor([1, 2, 3])}, [{"preds": 1}, {"preds": 2}, {"preds": 3}]),
    Case(
        {"preds": torch.tensor([1, 2, 3]), "metadata": torch.tensor([4, 5, 6])},
        [{"preds": 1, "metadata": 4}, {"preds": 2, "metadata": 5}, {"preds": 3, "metadata": 6}],
    ),
    Case(torch.tensor([1, 2, 3]), [torch.tensor(1), torch.tensor(2), torch.tensor(3)]),
    # Mixed
    Case(
        {"preds": torch.tensor([1, 2, 3]), "metadata": [4, 5, 6]},
        [{"preds": 1, "metadata": 4}, {"preds": 2, "metadata": 5}, {"preds": 3, "metadata": 6}],
    ),
]


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
@pytest.mark.parametrize("case", cases)
def test_default_uncollate(case):
    assert default_uncollate(case.collated_batch) == case.uncollated_batch


ErrorCase = namedtuple("ErrorCase", ["collated_batch", "match"])

error_cases = [
    ErrorCase({"preds": [1, 2, 3], "metadata": [4, 5, 6, 7]}, "expected to have the same length."),
    ErrorCase({"preds": [1, 2, 3], "metadata": "test"}, "expected to be list-like."),
    ErrorCase("test", "expected to be a `dict` or list-like"),
]


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
@pytest.mark.parametrize("error_case", error_cases)
def test_default_uncollate_raises(error_case):
    with pytest.raises(ValueError, match=error_case.match):
        default_uncollate(error_case.collated_batch)
