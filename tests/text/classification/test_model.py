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

from flash import Trainer
from flash.core.utilities.imports import _TEXT_AVAILABLE
from flash.text import TextClassifier

# ======== Mock functions ========


class DummyDataset(torch.utils.data.Dataset):

    def __getitem__(self, index):
        return {
            "input_ids": torch.randint(1000, size=(100, )),
            "labels": torch.randint(2, size=(1, )).item(),
        }

    def __len__(self) -> int:
        return 100


# ==============================

TEST_BACKBONE = "prajjwal1/bert-tiny"  # super small model for testing


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_AVAILABLE, reason="text libraries aren't installed.")
def test_init_train(tmpdir):
    model = TextClassifier(2, TEST_BACKBONE)
    train_dl = torch.utils.data.DataLoader(DummyDataset())
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model, train_dl)


@pytest.mark.skipif(not _TEXT_AVAILABLE, reason="text libraries aren't installed.")
def test_jit(tmpdir):
    sample_input = torch.randint(1000, size=(1, 100))
    path = os.path.join(tmpdir, "test.pt")

    model = TextClassifier(2, TEST_BACKBONE)
    model.eval()

    # Huggingface bert model only supports `torch.jit.trace` with `strict=False`
    model = torch.jit.trace(model, sample_input, strict=False)

    torch.jit.save(model, path)
    model = torch.jit.load(path)

    out = model(sample_input)["logits"]
    assert isinstance(out, torch.Tensor)
    assert out.shape == torch.Size([1, 2])
