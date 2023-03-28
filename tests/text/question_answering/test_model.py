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
import collections
from typing import Any

import pytest
import torch
from torch import Tensor

from flash.core.utilities.imports import _TOPIC_TEXT_AVAILABLE
from flash.text import QuestionAnsweringTask
from tests.helpers.task_tester import TaskTester

TEST_BACKBONE = "distilbert-base-uncased"


class TestQuestionAnsweringTask(TaskTester):
    task = QuestionAnsweringTask
    task_kwargs = {"backbone": TEST_BACKBONE}
    cli_command = "question_answering"
    is_testing = _TOPIC_TEXT_AVAILABLE
    is_available = _TOPIC_TEXT_AVAILABLE

    scriptable = False
    traceable = False

    @property
    def example_forward_input(self):
        return {
            "input_ids": torch.randint(1000, size=(1, 32)),
            "attention_mask": torch.randint(1, size=(1, 32)),
            "start_positions": torch.randint(1000, size=(1, 1)),
            "end_positions": torch.randint(1000, size=(1, 1)),
        }

    def check_forward_output(self, output: Any):
        assert isinstance(output[0], Tensor)
        assert isinstance(output[1], collections.OrderedDict)

    @property
    def example_train_sample(self):
        return {
            "question": "A question",
            "answer": {"text": ["The answer"], "answer_start": [0]},
            "context": "The paragraph of text which contains the answer to the question",
            "id": 0,
        }

    @property
    def example_val_sample(self):
        return self.example_train_sample

    @property
    def example_test_sample(self):
        return self.example_train_sample


@pytest.mark.skipif(not _TOPIC_TEXT_AVAILABLE, reason="text libraries aren't installed.")
def test_modules_to_freeze():
    model = QuestionAnsweringTask(backbone=TEST_BACKBONE)
    assert model.modules_to_freeze() is model.model.distilbert
