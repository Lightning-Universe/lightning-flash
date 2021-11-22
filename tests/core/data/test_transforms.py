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
from torch import nn

from flash.core.data.io.input import DataKeys
from flash.core.data.transforms import ApplyToKeys, kornia_collate, KorniaParallelTransforms, merge_transforms
from flash.core.data.utils import convert_to_modules


class TestApplyToKeys:
    @pytest.mark.parametrize(
        "sample, keys, expected",
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

    @pytest.mark.parametrize(
        "transform, expected",
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


@pytest.mark.parametrize("with_params", [True, False])
def test_kornia_parallel_transforms(with_params):
    samples = [torch.rand(1, 3, 10, 10), torch.rand(1, 3, 10, 10)]
    transformed_sample = torch.rand(1, 3, 10, 10)

    transform_a = Mock(spec=torch.nn.Module, return_value=transformed_sample)
    transform_b = Mock(spec=torch.nn.Module)

    if with_params:
        transform_a._params = "test"  # initialize params with some value

    parallel_transforms = KorniaParallelTransforms(transform_a, transform_b)
    parallel_transforms(samples)

    assert transform_a.call_count == 2
    assert transform_b.call_count == 2

    if with_params:
        assert transform_a.call_args_list[1][0][1] == "test"
        # check that after the forward `_params` is set to None
        assert transform_a._params == transform_a._params is None

    assert torch.allclose(transform_a.call_args_list[0][0][0], samples[0])
    assert torch.allclose(transform_a.call_args_list[1][0][0], samples[1])
    assert torch.allclose(transform_b.call_args_list[0][0][0], transformed_sample)
    assert torch.allclose(transform_b.call_args_list[1][0][0], transformed_sample)


def test_kornia_collate():
    samples = [
        {DataKeys.INPUT: torch.zeros(1, 3, 10, 10), DataKeys.TARGET: 1},
        {DataKeys.INPUT: torch.zeros(1, 3, 10, 10), DataKeys.TARGET: 2},
        {DataKeys.INPUT: torch.zeros(1, 3, 10, 10), DataKeys.TARGET: 3},
    ]

    result = kornia_collate(samples)
    assert torch.all(result[DataKeys.TARGET] == torch.tensor([1, 2, 3]))
    assert list(result[DataKeys.INPUT].shape) == [3, 3, 10, 10]
    assert torch.allclose(result[DataKeys.INPUT], torch.zeros(1))


_MOCK_TRANSFORM = Mock()


@pytest.mark.parametrize(
    "base_transforms, additional_transforms, expected_result",
    [
        (
            {"to_tensor_transform": _MOCK_TRANSFORM},
            {"post_tensor_transform": _MOCK_TRANSFORM},
            {"to_tensor_transform": _MOCK_TRANSFORM, "post_tensor_transform": _MOCK_TRANSFORM},
        ),
        (
            {"to_tensor_transform": _MOCK_TRANSFORM},
            {"to_tensor_transform": _MOCK_TRANSFORM},
            {
                "to_tensor_transform": nn.Sequential(
                    convert_to_modules(_MOCK_TRANSFORM), convert_to_modules(_MOCK_TRANSFORM)
                )
            },
        ),
        (
            {"to_tensor_transform": _MOCK_TRANSFORM},
            {"to_tensor_transform": _MOCK_TRANSFORM, "post_tensor_transform": _MOCK_TRANSFORM},
            {
                "to_tensor_transform": nn.Sequential(
                    convert_to_modules(_MOCK_TRANSFORM), convert_to_modules(_MOCK_TRANSFORM)
                ),
                "post_tensor_transform": _MOCK_TRANSFORM,
            },
        ),
        (
            {"to_tensor_transform": _MOCK_TRANSFORM, "post_tensor_transform": _MOCK_TRANSFORM},
            {"to_tensor_transform": _MOCK_TRANSFORM},
            {
                "to_tensor_transform": nn.Sequential(
                    convert_to_modules(_MOCK_TRANSFORM), convert_to_modules(_MOCK_TRANSFORM)
                ),
                "post_tensor_transform": _MOCK_TRANSFORM,
            },
        ),
    ],
)
def test_merge_transforms(base_transforms, additional_transforms, expected_result):
    result = merge_transforms(base_transforms, additional_transforms)
    assert result.keys() == expected_result.keys()
    for key in result:
        if result[key] == _MOCK_TRANSFORM:
            assert expected_result[key] == _MOCK_TRANSFORM
        elif isinstance(result[key], nn.Sequential):
            assert isinstance(expected_result[key], nn.Sequential)
            assert len(result[key]) == len(expected_result[key])
            for module, expected_module in zip(result[key], expected_result[key]):
                assert module.func == expected_module.func
