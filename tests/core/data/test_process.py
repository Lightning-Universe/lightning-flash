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

from flash.core.data.data_module import DataModule
from flash.core.data.io.input_transform import DefaultInputTransform
from flash.core.data.data_source import DefaultDataSources


class CustomInputTransform(DefaultInputTransform):
    def __init__(self):
        super().__init__(
            data_sources={
                "test": Mock(return_value="test"),
                DefaultDataSources.TENSORS: Mock(return_value="tensors"),
            },
            default_data_source="test",
        )


def test_data_source_of_name():
    input_transform = CustomInputTransform()

    assert input_transform.data_source_of_name("test")() == "test"
    assert input_transform.data_source_of_name(DefaultDataSources.TENSORS)() == "tensors"
    assert input_transform.data_source_of_name("tensors")() == "tensors"
    assert input_transform.data_source_of_name("default")() == "test"

    with pytest.raises(MisconfigurationException, match="available data sources are: test, tensor"):
        input_transform.data_source_of_name("not available")


def test_available_data_sources():
    input_transform = CustomInputTransform()

    assert DefaultDataSources.TENSORS in input_transform.available_data_sources()
    assert "test" in input_transform.available_data_sources()
    assert len(input_transform.available_data_sources()) == 3

    data_module = DataModule(input_transform=input_transform)

    assert DefaultDataSources.TENSORS in data_module.available_data_sources()
    assert "test" in data_module.available_data_sources()
    assert len(data_module.available_data_sources()) == 3


def test_check_transforms():
    transform = torch.nn.Identity()
    DefaultInputTransform(train_transform=transform)
    DefaultInputTransform(train_transform=[transform])
