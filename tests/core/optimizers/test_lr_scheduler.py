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
import math

import pytest
from flash.core.optimizers import LinearWarmupCosineAnnealingLR
from flash.core.utilities.imports import _TOPIC_CORE_AVAILABLE
from torch import nn
from torch.optim import Adam


@pytest.mark.skipif(not _TOPIC_CORE_AVAILABLE, reason="Not testing core.")
@pytest.mark.parametrize(
    ("lr", "warmup_epochs", "max_epochs", "warmup_start_lr", "eta_min"),
    [
        (1, 10, 3200, 0.001, 0.0),
        (1e-4, 40, 300, 1e-6, 1e-5),
        (0.01, 1, 10, 0.0, 0.0),
        (0.01, 0, 10, 0.0, 0.0),  # only cosine decay
        (0.01, 10, 10, 0.0, 0.0),  # only linear warmup
    ],
)
def test_linear_warmup_cosine_annealing_lr(tmpdir, lr, warmup_epochs, max_epochs, warmup_start_lr, eta_min):
    layer1 = nn.Linear(10, 1)
    layer2 = nn.Linear(10, 1)
    optimizer1 = Adam(layer1.parameters(), lr=lr)
    optimizer2 = Adam(layer2.parameters(), lr=lr)

    scheduler1 = LinearWarmupCosineAnnealingLR(
        optimizer1,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        warmup_start_lr=warmup_start_lr,
        eta_min=eta_min,
    )

    scheduler2 = LinearWarmupCosineAnnealingLR(
        optimizer2,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        warmup_start_lr=warmup_start_lr,
        eta_min=eta_min,
    )

    # compares closed form lr values against values of get_lr function
    for epoch in range(max_epochs):
        scheduler1.step(epoch)
        expected_lr = scheduler1.get_last_lr()[0]
        current_lr = scheduler2.get_last_lr()[0]

        assert math.isclose(expected_lr, current_lr, rel_tol=1e-12)
        optimizer1.step()
        optimizer2.step()
        scheduler2.step()
