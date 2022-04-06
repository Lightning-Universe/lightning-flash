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
import re
from unittest import mock

import pytest
import torch

from flash import Trainer
from flash.__main__ import main
from flash.core.utilities.imports import _TEXT_AVAILABLE, _TEXT_TESTING
from flash.text import QuestionAnsweringTask

# ======== Mock functions ========

SEQUENCE_LENGTH = 384


class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        return {
            "input_ids": torch.randint(1000, size=(SEQUENCE_LENGTH,)),
            "attention_mask": torch.randint(1, size=(SEQUENCE_LENGTH,)),
            "start_positions": torch.randint(1000, size=(1,)),
            "end_positions": torch.randint(1000, size=(1,)),
        }

    def __len__(self) -> int:
        return 100


# ==============================

TEST_BACKBONE = "distilbert-base-uncased"


def test_modules_to_freeze():
    model = QuestionAnsweringTask(backbone=TEST_BACKBONE)
    assert model.modules_to_freeze() is model.model.distilbert


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
def test_init_train(tmpdir):
    model = QuestionAnsweringTask(TEST_BACKBONE)
    train_dl = torch.utils.data.DataLoader(DummyDataset())
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model, train_dl)


@pytest.mark.skipif(_TEXT_AVAILABLE, reason="text libraries are installed.")
def test_load_from_checkpoint_dependency_error():
    with pytest.raises(ModuleNotFoundError, match=re.escape("'lightning-flash[text]'")):
        QuestionAnsweringTask.load_from_checkpoint("not_a_real_checkpoint.pt")


@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
def test_cli():
    cli_args = ["flash", "question_answering", "--trainer.fast_dev_run", "True"]
    with mock.patch("sys.argv", cli_args):
        try:
            main()
        except SystemExit:
            pass
