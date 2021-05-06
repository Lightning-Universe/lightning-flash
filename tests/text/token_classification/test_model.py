import os

import pytest
import torch

from flash import Trainer
from flash.text import TokenClassifier

TEST_BACKBONE = "prajjwal1/bert-tiny"  # super small model for testing
NUM_LABELS = 10

# ======== Mock functions ========


class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        return {
            "input_ids": torch.randint(1000, size=(100,)),
            "labels": torch.randint(NUM_LABELS, size=(100,)),
        }

    def __len__(self) -> int:
        return 100


# ==============================


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
def test_init_train(tmpdir):
    model = TokenClassifier(NUM_LABELS, TEST_BACKBONE)
    train_dl = torch.utils.data.DataLoader(DummyDataset())
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model, train_dl)
