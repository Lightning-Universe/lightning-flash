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
from unittest import mock

import pytest
import torch
from torch import Tensor

from flash import DataKeys
from flash.core.utilities.imports import _SERVE_TESTING, _TEXT_TESTING, _TOPIC_TEXT_AVAILABLE
from flash.text import SummarizationTask
from tests.helpers.task_tester import TaskTester

TEST_BACKBONE = "sshleifer/tiny-mbart"  # tiny model for testing


class TestSummarizationTask(TaskTester):
    task = SummarizationTask
    task_kwargs = {
        "backbone": TEST_BACKBONE,
        "tokenizer_kwargs": {"src_lang": "en_XX", "tgt_lang": "en_XX"},
    }
    cli_command = "summarization"
    is_testing = _TEXT_TESTING
    is_available = _TOPIC_TEXT_AVAILABLE

    scriptable = False

    @property
    def example_forward_input(self):
        return {
            "input_ids": torch.randint(128, size=(1, 32)),
        }

    def check_forward_output(self, output: Any):
        assert isinstance(output, Tensor)
        assert output.shape == torch.Size([1, 128])

    @property
    def example_train_sample(self):
        return {DataKeys.INPUT: "Some long passage of text", DataKeys.TARGET: "A summary"}

    @property
    def example_val_sample(self):
        return self.example_train_sample

    @property
    def example_test_sample(self):
        return self.example_train_sample


@pytest.mark.skipif(not _SERVE_TESTING, reason="serve libraries aren't installed.")
@mock.patch("flash._IS_TESTING", True)
def test_serve():
    model = SummarizationTask(TEST_BACKBONE)
    model.eval()
    model.serve()
