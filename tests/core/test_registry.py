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


def test_registry(tmpdir):

    backbones = FlashRegistry("backbones")

    @backbones.register_function()
    def my_model(nc_input=5, nc_output=6):
        return nn.Linear(nc_input, nc_output), nc_input, nc_output

    mlp, nc_input, nc_output = backbones.get("my_model")(nc_output=7)
    assert nc_input == 5
    assert nc_output == 7
    assert mlp.weight.shape == torch.Size([7, 5])

    backbones.register_function(my_model, name="cho")
    assert backbones.get("cho")

    with pytest.raises(MisconfigurationException, match="``register_function`` should be used with a function"):
        backbones.register_function(nn.Linear(1, 1), name="cho")

    backbones.register_function(my_model, name="cho", override=True)

    with pytest.raises(MisconfigurationException, match="Name cho is already present within"):
        backbones.register_function(my_model, name="cho", override=False)
