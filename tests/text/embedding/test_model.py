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

import pytest
import torch

import flash
from flash.text import TextClassificationData, TextEmbedder
from tests.helpers.utils import _TEXT_TESTING

# ======== Mock data ========

predict_data = [
    "Turgid dialogue, feeble characterization - Harvey Keitel a judge?.",
    "The worst movie in the history of cinema.",
    "I come from Bulgaria where it 's almost impossible to have a tornado.",
]
# ==============================

TEST_BACKBONE = "sentence-transformers/all-MiniLM-L6-v2"  # super small model for testing
model = TextEmbedder(backbone=TEST_BACKBONE)


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
def test_predict(tmpdir):
    datamodule = TextClassificationData.from_lists(predict_data=predict_data)
    trainer = flash.Trainer(gpus=torch.cuda.device_count())
    predictions = trainer.predict(model, datamodule=datamodule)
    assert [t.size() for t in predictions[0]] == [torch.Size([384]), torch.Size([384]), torch.Size([384])]
