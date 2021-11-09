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
from flash.core.data.data_pipeline import DefaultPreprocess
from flash.core.data.data_source import DefaultDataSources


class CustomPreprocess(DefaultPreprocess):
    def __init__(self):
        super().__init__(
            data_sources={
                "test": Mock(return_value="test"),
                DefaultDataSources.TENSORS: Mock(return_value="tensors"),
            },
            default_data_source="test",
        )


def test_data_source_of_name():
    preprocess = CustomPreprocess()

    assert preprocess.data_source_of_name("test")() == "test"
    assert preprocess.data_source_of_name(DefaultDataSources.TENSORS)() == "tensors"
    assert preprocess.data_source_of_name("tensors")() == "tensors"
    assert preprocess.data_source_of_name("default")() == "test"

    with pytest.raises(MisconfigurationException, match="available data sources are: test, tensor"):
        preprocess.data_source_of_name("not available")


def test_available_data_sources():
    preprocess = CustomPreprocess()

    assert DefaultDataSources.TENSORS in preprocess.available_data_sources()
    assert "test" in preprocess.available_data_sources()
    assert len(preprocess.available_data_sources()) == 3

    data_module = DataModule(preprocess=preprocess)

    assert DefaultDataSources.TENSORS in data_module.available_data_sources()
    assert "test" in data_module.available_data_sources()
    assert len(data_module.available_data_sources()) == 3


def test_check_transforms():
    transform = torch.nn.Identity()
    DefaultPreprocess(train_transform=transform)
    DefaultPreprocess(train_transform=[transform])
