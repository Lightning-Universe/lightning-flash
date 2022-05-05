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
import os
from typing import Any

import pytest
import torch

import flash
from flash.core.utilities.imports import _TEXT_AVAILABLE, _TEXT_TESTING
from flash.text import TextClassificationData, TextEmbedder
from tests.helpers.task_tester import TaskTester

# ======== Mock data ========

predict_data = [
    "Turgid dialogue, feeble characterization - Harvey Keitel a judge?.",
    "The worst movie in the history of cinema.",
    "I come from Bulgaria where it 's almost impossible to have a tornado.",
]

# ==============================

TEST_BACKBONE = "sentence-transformers/all-MiniLM-L6-v2"  # tiny model for testing


class TestTextEmbedder(TaskTester):

    task = TextEmbedder
    task_kwargs = {"backbone": TEST_BACKBONE}
    is_testing = _TEXT_TESTING
    is_available = _TEXT_AVAILABLE

    scriptable = False

    @property
    def example_forward_input(self):
        return {"input_ids": torch.randint(1000, size=(1, 100))}

    def check_forward_output(self, output: Any):
        assert isinstance(output, torch.Tensor)
        assert output.shape == torch.Size([1, 384])


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
def test_predict(tmpdir):
    datamodule = TextClassificationData.from_lists(predict_data=predict_data, batch_size=4)
    model = TextEmbedder(backbone=TEST_BACKBONE)

    trainer = flash.Trainer(gpus=torch.cuda.device_count())
    predictions = trainer.predict(model, datamodule=datamodule)
    assert [t.size() for t in predictions[0]] == [torch.Size([384]), torch.Size([384]), torch.Size([384])]
