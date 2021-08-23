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
from torch import nn

from flash.core.optimizers import LAMB, LARS, LinearWarmupCosineAnnealingLR


@pytest.mark.parametrize(
    "optim_fn, lr, kwargs",
    [
        (LARS, 0.1, {}),
        (LARS, 0.1, {"weight_decay": 0.001}),
        (LARS, 0.1, {"momentum": 0.9}),
        (LAMB, 1e-3, {}),
        (LAMB, 1e-3, {"amsgrad": True}),
        (LAMB, 1e-3, {"weight_decay": 0.001}),
    ],
)
def test_optim_call(tmpdir, optim_fn, lr, kwargs):
    layer = nn.Linear(10, 1)
    optimizer = optim_fn(layer.parameters(), lr=lr, **kwargs)

    for _ in range(10):
        dummy_input = torch.rand(1, 10)
        dummy_input.requires_grad = True
        result = layer(dummy_input)
        result.backward()
        optimizer.step()


@pytest.mark.parametrize("optim_fn, lr", [(LARS, 0.1), (LAMB, 1e-3)])
def test_optim_with_scheduler(tmpdir, optim_fn, lr):
    max_epochs = 10
    layer = nn.Linear(10, 1)
    optimizer = optim_fn(layer.parameters(), lr=lr)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=2, max_epochs=max_epochs)

    for _ in range(max_epochs):
        dummy_input = torch.rand(1, 10)
        dummy_input.requires_grad = True
        result = layer(dummy_input)
        result.backward()
        optimizer.step()
        scheduler.step()
