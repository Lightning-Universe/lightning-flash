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
import pytest
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn

from flash.core.registry import FlashRegistry


def test_registry_raises():
    backbones = FlashRegistry("backbones")

    @backbones()
    def my_model(nc_input=5, nc_output=6):
        return nn.Linear(nc_input, nc_output), nc_input, nc_output

    with pytest.raises(MisconfigurationException, match="You can only register a function, found: Linear"):
        backbones(nn.Linear(1, 1), name="cho")

    backbones(my_model, name="cho", override=True)
    with pytest.raises(MisconfigurationException, match="Function with name: cho and metadata: {}"):
        backbones(my_model, name="cho", override=False)

    backbones.remove("cho")
    with pytest.raises(KeyError, match="Key: cho is not in FlashRegistry"):
        backbones.get("cho")


def test_registry():
    backbones = FlashRegistry("backbones")

    @backbones()
    def my_model(nc_input=5, nc_output=6):
        return nn.Linear(nc_input, nc_output), nc_input, nc_output

    mlp, nc_input, nc_output = backbones.get("my_model")(nc_output=7)
    assert nc_input == 5
    assert nc_output == 7
    assert mlp.weight.shape == (7, 5)

    backbones(my_model, name="cho")
    assert backbones.get("cho")
    backbones.remove("cho")

    backbones(my_model, name="cho", namespace="timm")
    function = backbones.get("cho", with_metadata=True, strict=False)
    assert function["metadata"] == {"namespace": "timm"}

    backbones(my_model, name="cho", namespace="timm", type="resnet")
    backbones(my_model, name="cho", namespace="torchvision", type="resnet")
    backbones(my_model, name="cho", namespace="timm", type="densenet")
    backbones(my_model, name="cho", namespace="timm", type="alexnet")

    function = backbones.get("cho", with_metadata=True, type="resnet", namespace="timm")
    assert function["name"] == "cho"
    assert function["metadata"] == {"namespace": "timm", "type": "resnet"}

    functions = backbones.get("cho", with_metadata=True, namespace="timm", strict=False)
    assert len(functions) == 4
    assert backbones.available_keys() == ['cho', 'cho', 'cho', 'cho', 'cho', 'my_model']
