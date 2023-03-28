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
import time

import pytest
import torch
from pytorch_lightning import LightningModule
from torch import nn

from flash.core.utilities.embedder import Embedder
from flash.core.utilities.imports import _TOPIC_CORE_AVAILABLE


class EmbedderTestModel(LightningModule):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        return self.backbone(batch)


class NLayerModel(EmbedderTestModel):
    def __init__(self, n_layers):
        super().__init__(nn.Sequential(*[nn.Linear(1000, 1000) for _ in range(n_layers)]))


@pytest.mark.skipif(not _TOPIC_CORE_AVAILABLE, reason="Not testing core.")
@pytest.mark.parametrize("layer, size", [("backbone.1", 30), ("output", 40), ("", 40)])
def test_embedder(layer, size):
    """Tests that the embedder ``predict_step`` correctly returns the output from the requested layer."""
    model = EmbedderTestModel(
        nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 30),
            nn.Linear(30, 40),
        )
    )

    embedder = Embedder(model, layer)

    assert embedder.predict_step(torch.rand(10, 10), 0, 0).size(1) == size
    assert embedder(torch.rand(10, 10)).size(1) == size


@pytest.mark.skipif(not _TOPIC_CORE_AVAILABLE, reason="Not testing core.")
def test_embedder_scaling_overhead():
    """Tests that embedding to the 3rd layer of a 200 layer model takes less than double the time of embedding to.

    the same layer of a 3 layer model and therefore in the order of 10s - 100s of times faster than executing the full
    200 layer model.

    Note that this bound is intentionally high in an effort to reduce the flakiness of the test.
    """
    shallow_embedder = Embedder(NLayerModel(3), "backbone.2")

    start = time.perf_counter()
    shallow_embedder.predict_step(torch.rand(10, 1000), 0, 0)
    end = time.perf_counter()

    shallow_time = end - start

    deep_embedder = Embedder(NLayerModel(200), "backbone.2")

    start = time.perf_counter()
    deep_embedder.predict_step(torch.rand(10, 1000), 0, 0)
    end = time.perf_counter()

    deep_time = end - start

    assert (abs(deep_time - shallow_time) / shallow_time) < 1


@pytest.mark.skipif(not _TOPIC_CORE_AVAILABLE, reason="Not testing core.")
def test_embedder_raising_overhead():
    """Tests that embedding to the output layer of a 3 layer model takes less than 10ms more than the time taken to
    execute the model without the embedder.

    Note that this bound is intentionally high in an effort to reduce the flakiness of the test.
    """
    model = NLayerModel(10)
    embedder = Embedder(model, "output")

    start = time.perf_counter()
    model.predict_step(torch.rand(10, 1000), 0, 0)
    end = time.perf_counter()

    model_time = end - start

    start = time.perf_counter()
    embedder.predict_step(torch.rand(10, 1000), 0, 0)
    end = time.perf_counter()

    embedder_time = end - start

    assert abs(embedder_time - model_time) < 0.01
