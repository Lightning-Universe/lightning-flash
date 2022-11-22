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
from typing import Any

import pytest
import torch
from torch import Tensor

from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _IMAGE_AVAILABLE, _IMAGE_TESTING
from flash.image.style_transfer import StyleTransfer
from tests.helpers.task_tester import TaskTester


class TestStyleTransfer(TaskTester):

    task = StyleTransfer
    cli_command = "style_transfer"
    is_testing = _IMAGE_TESTING
    is_available = _IMAGE_AVAILABLE

    # TODO: loss_fn and perceptual_loss can't be jitted
    scriptable = False
    traceable = False

    @property
    def example_forward_input(self):
        return torch.rand(1, 3, 32, 32)

    def check_forward_output(self, output: Any):
        assert isinstance(output, Tensor)
        assert output.shape == torch.Size([1, 3, 32, 32])

    @property
    def example_train_sample(self):
        return {DataKeys.INPUT: torch.rand(3, 224, 224)}


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_style_transfer_task():
    model = StyleTransfer(
        backbone="vgg11", content_layer="relu1_2", content_weight=10, style_layers="relu1_2", style_weight=11
    )
    assert model.perceptual_loss.content_loss.encoder.layer == "relu1_2"
    assert model.perceptual_loss.content_loss.score_weight == 10
    assert "relu1_2" in [n for n, m in model.perceptual_loss.style_loss.named_modules()]
    assert model.perceptual_loss.style_loss.score_weight == 11
