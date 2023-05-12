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
import torch

from flash.core.data.io.input import DataKeys
from flash.core.data.transforms import ApplyToKeys
from flash.core.utilities.imports import _TOPIC_CORE_AVAILABLE


class TestApplyToKeys:
    @pytest.mark.skipif(not _TOPIC_CORE_AVAILABLE, reason="Not testing core.")
    @pytest.mark.parametrize(
        ("sample", "keys", "expected"),
        [
            ({DataKeys.INPUT: "test"}, DataKeys.INPUT, "test"),
            (
                {DataKeys.INPUT: "test_a", DataKeys.TARGET: "test_b"},
                [DataKeys.INPUT, DataKeys.TARGET],
                ["test_a", "test_b"],
            ),
            ({"input": "test"}, "input", "test"),
            ({"input": "test_a", "target": "test_b"}, ["input", "target"], ["test_a", "test_b"]),
            ({"input": "test_a", "target": "test_b", "extra": "..."}, ["input", "target"], ["test_a", "test_b"]),
            ({"input": "test_a", "target": "test_b"}, ["input", "target", "extra"], ["test_a", "test_b"]),
            ({"target": "..."}, "input", None),
        ],
    )
    def test_forward(self, sample, keys, expected):
        transform = Mock(return_value=["out"] * len(keys))
        ApplyToKeys(keys, transform)(sample)
        if expected is not None:
            transform.assert_called_once_with(expected)
        else:
            transform.assert_not_called()

    @pytest.mark.skipif(not _TOPIC_CORE_AVAILABLE, reason="Not testing core.")
    @pytest.mark.parametrize(
        ("transform", "expected"),
        [
            (
                ApplyToKeys(DataKeys.INPUT, torch.nn.ReLU()),
                "ApplyToKeys(keys=<DataKeys.INPUT: 'input'>, transform=ReLU())",
            ),
            (
                ApplyToKeys([DataKeys.INPUT, DataKeys.TARGET], torch.nn.ReLU()),
                "ApplyToKeys(keys=[<DataKeys.INPUT: 'input'>, " "<DataKeys.TARGET: 'target'>], transform=ReLU())",
            ),
            (ApplyToKeys("input", torch.nn.ReLU()), "ApplyToKeys(keys='input', transform=ReLU())"),
            (
                ApplyToKeys(["input", "target"], torch.nn.ReLU()),
                "ApplyToKeys(keys=['input', 'target'], transform=ReLU())",
            ),
        ],
    )
    def test_repr(self, transform, expected):
        assert repr(transform) == expected
