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
from torch import nn

from flash.core.optimizers import LARS, LAMB, LinearWarmupCosineAnnealingLR


@pytest.mark.parametrize("optim_fn, lr", [(LARS, 0.1), (LAMB, 1e-3)])
def test_optim_call(tmpdir, optim_fn, lr):
    layer = nn.Linear(10, 1)
    optimizer = optim_fn(layer.parameters(), lr=lr)

    for _ in range(10):
        optimizer.step()


@pytest.mark.parametrize("optim_fn, lr", [(LARS, 0.1), (LAMB, 1e-3)])
def test_optim_with_scheduler(tmpdir, optim_fn, lr):
    max_epochs = 10
    layer = nn.Linear(10, 1)
    optimizer = optim_fn(layer.parameters(), lr=lr)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=2, max_epochs=max_epochs)

    for _ in range(max_epochs):
        optimizer.step()
        scheduler.step()
