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
from unittest.mock import ANY, Mock

import pytest
import torch
from torch import nn

from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.transforms import kornia_collate, KorniaParallelTransforms, merge_transforms
from flash.core.data.utils import convert_to_modules


@pytest.mark.parametrize("with_params", [True, False])
def test_kornia_parallel_transforms(with_params):
    sample = torch.rand(1, 3, 10, 10)
    transformed_sample = torch.rand(1, 3, 10, 10)

    transform_a = Mock(spec=torch.nn.Module, return_value=transformed_sample)
    transform_b = Mock(spec=torch.nn.Module)

    if with_params:
        transform_a._params = "test"

    parallel_transforms = KorniaParallelTransforms(transform_a, transform_b)

    parallel_transforms(sample)

    if with_params:
        transform_a.assert_called_once()
        transform_b.assert_called_once()

        assert transform_a.call_args[0][1] == "test"
    else:
        transform_a.assert_called_once()
        transform_b.assert_called_once()

    assert torch.allclose(transform_a.call_args[0][0], sample)
    assert torch.allclose(transform_b.call_args[0][0], transformed_sample)


def test_kornia_collate():
    samples = [
        {
            DefaultDataKeys.INPUT: torch.zeros(1, 3, 10, 10),
            DefaultDataKeys.TARGET: 1
        },
        {
            DefaultDataKeys.INPUT: torch.zeros(1, 3, 10, 10),
            DefaultDataKeys.TARGET: 2
        },
        {
            DefaultDataKeys.INPUT: torch.zeros(1, 3, 10, 10),
            DefaultDataKeys.TARGET: 3
        },
    ]

    result = kornia_collate(samples)
    assert torch.all(result[DefaultDataKeys.TARGET] == torch.tensor([1, 2, 3]))
    assert list(result[DefaultDataKeys.INPUT].shape) == [3, 3, 10, 10]
    assert torch.allclose(result[DefaultDataKeys.INPUT], torch.zeros(1))


_MOCK_TRANSFORM = Mock()


@pytest.mark.parametrize(
    "base_transforms, additional_transforms, expected_result",
    [
        (
            {
                "to_tensor_transform": _MOCK_TRANSFORM
            },
            {
                "post_tensor_transform": _MOCK_TRANSFORM
            },
            {
                "to_tensor_transform": _MOCK_TRANSFORM,
                "post_tensor_transform": _MOCK_TRANSFORM
            },
        ),
        (
            {
                "to_tensor_transform": _MOCK_TRANSFORM
            },
            {
                "to_tensor_transform": _MOCK_TRANSFORM
            },
            {
                "to_tensor_transform": nn.Sequential(
                    convert_to_modules(_MOCK_TRANSFORM), convert_to_modules(_MOCK_TRANSFORM)
                )
            },
        ),
        (
            {
                "to_tensor_transform": _MOCK_TRANSFORM
            },
            {
                "to_tensor_transform": _MOCK_TRANSFORM,
                "post_tensor_transform": _MOCK_TRANSFORM
            },
            {
                "to_tensor_transform": nn.Sequential(
                    convert_to_modules(_MOCK_TRANSFORM), convert_to_modules(_MOCK_TRANSFORM)
                ),
                "post_tensor_transform": _MOCK_TRANSFORM
            },
        ),
        (
            {
                "to_tensor_transform": _MOCK_TRANSFORM,
                "post_tensor_transform": _MOCK_TRANSFORM
            },
            {
                "to_tensor_transform": _MOCK_TRANSFORM
            },
            {
                "to_tensor_transform": nn.Sequential(
                    convert_to_modules(_MOCK_TRANSFORM), convert_to_modules(_MOCK_TRANSFORM)
                ),
                "post_tensor_transform": _MOCK_TRANSFORM
            },
        ),
    ],
)
def test_merge_transforms(base_transforms, additional_transforms, expected_result):
    result = merge_transforms(base_transforms, additional_transforms)
    assert result.keys() == expected_result.keys()
    for key in result.keys():
        if result[key] == _MOCK_TRANSFORM:
            assert expected_result[key] == _MOCK_TRANSFORM
        elif isinstance(result[key], nn.Sequential):
            assert isinstance(expected_result[key], nn.Sequential)
            assert len(result[key]) == len(expected_result[key])
            for module, expected_module in zip(result[key], expected_result[key]):
                assert module.func == expected_module.func
