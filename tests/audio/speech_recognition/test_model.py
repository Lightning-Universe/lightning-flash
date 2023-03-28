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
# limitations under the License.import os
from typing import Any
from unittest import mock

import numpy as np
import pytest
import torch
from torch import Tensor

from flash.audio import SpeechRecognition
from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _AUDIO_TESTING, _SERVE_TESTING, _TOPIC_AUDIO_AVAILABLE
from tests.helpers.task_tester import TaskTester

TEST_BACKBONE = "patrickvonplaten/wav2vec2_tiny_random_robust"  # tiny model for testing


class TestSpeechRecognition(TaskTester):
    task = SpeechRecognition
    task_kwargs = dict(backbone=TEST_BACKBONE)
    cli_command = "speech_recognition"
    is_testing = _AUDIO_TESTING
    is_available = _TOPIC_AUDIO_AVAILABLE

    scriptable = False

    @property
    def example_forward_input(self):
        return {"input_values": torch.randn(size=torch.Size([1, 86631])).float()}

    def check_forward_output(self, output: Any):
        assert isinstance(output, Tensor)
        assert output.shape == torch.Size([1, 95, 12])

    @property
    def example_train_sample(self):
        return {
            DataKeys.INPUT: np.random.randn(86631),
            DataKeys.TARGET: "some target text",
            DataKeys.METADATA: {"sampling_rate": 16000},
        }

    @property
    def example_val_sample(self):
        return self.example_train_sample

    @property
    def example_test_sample(self):
        return self.example_train_sample


@pytest.mark.skipif(not _AUDIO_TESTING, reason="audio libraries aren't installed.")
def test_modules_to_freeze():
    model = SpeechRecognition(backbone=TEST_BACKBONE)
    assert model.modules_to_freeze() is model.model.wav2vec2


@pytest.mark.skipif(not _SERVE_TESTING, reason="serve libraries aren't installed.")
@mock.patch("flash._IS_TESTING", True)
def test_serve():
    model = SpeechRecognition(backbone=TEST_BACKBONE)
    model.eval()
    model.serve()
