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
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data._utils.collate import default_collate

from flash import DataModule
from flash.core.data.io.input import InputFormat
from flash.core.data.io.input_transform import (
    _InputTransformProcessor,
    _InputTransformSequential,
    DefaultInputTransform,
)
from flash.core.utilities.stages import RunningStage


class CustomInputTransform(DefaultInputTransform):
    def __init__(self):
        super().__init__(
            inputs={
                "test": Mock(return_value="test"),
                InputFormat.TENSORS: Mock(return_value="tensors"),
            },
            default_input="test",
        )


def test_input_transform_processor_str():
    input_transform_processor = _InputTransformProcessor(
        Mock(name="input_transform"),
        default_collate,
        torch.relu,
        torch.softmax,
        RunningStage.TRAINING,
        False,
        True,
    )
    assert str(input_transform_processor) == (
        "_InputTransformProcessor:\n"
        "\t(per_sample_transform): FuncModule(relu)\n"
        "\t(collate_fn): FuncModule(default_collate)\n"
        "\t(per_batch_transform): FuncModule(softmax)\n"
        "\t(apply_per_sample_transform): False\n"
        "\t(on_device): True\n"
        "\t(stage): RunningStage.TRAINING"
    )


def test_sequential_str():
    sequential = _InputTransformSequential(
        Mock(name="input_transform"),
        torch.softmax,
        torch.as_tensor,
        torch.relu,
        RunningStage.TRAINING,
        True,
    )
    assert str(sequential) == (
        "_InputTransformSequential:\n"
        "\t(pre_tensor_transform): FuncModule(softmax)\n"
        "\t(to_tensor_transform): FuncModule(as_tensor)\n"
        "\t(post_tensor_transform): FuncModule(relu)\n"
        "\t(assert_contains_tensor): True\n"
        "\t(stage): RunningStage.TRAINING"
    )


def test_input_of_name():
    input_transform = CustomInputTransform()

    assert input_transform.input_of_name("test")() == "test"
    assert input_transform.input_of_name(InputFormat.TENSORS)() == "tensors"
    assert input_transform.input_of_name("tensors")() == "tensors"
    assert input_transform.input_of_name("default")() == "test"

    with pytest.raises(MisconfigurationException, match="available data sources are: test, tensor"):
        input_transform.input_of_name("not available")


def test_available_inputs():
    input_transform = CustomInputTransform()

    assert InputFormat.TENSORS in input_transform.available_inputs()
    assert "test" in input_transform.available_inputs()
    assert len(input_transform.available_inputs()) == 3

    data_module = DataModule(input_transform=input_transform)

    assert InputFormat.TENSORS in data_module.available_inputs()
    assert "test" in data_module.available_inputs()
    assert len(data_module.available_inputs()) == 3


def test_check_transforms():
    transform = torch.nn.Identity()
    DefaultInputTransform(train_transform=transform)
    DefaultInputTransform(train_transform=[transform])
