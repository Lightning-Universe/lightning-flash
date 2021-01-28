import os

import torch

from flash import Trainer
from flash.text import TextClassifier

# ======== Mock functions ========


class DummyDataset(torch.utils.data.Dataset):

    def __getitem__(self, index):
        return {
            "input_ids": torch.randint(1000, size=(100, )),
            "labels": torch.randint(2, size=(1, )).item(),
        }

    def __len__(self):
        return 100


# ==============================

TEST_BACKBONE = "prajjwal1/bert-tiny"  # super small model for testing


def test_init_train(tmpdir):
    if os.name == "nt":
        # TODO: huggingface stuff timing out on windows
        #
        return True
    model = TextClassifier(2, TEST_BACKBONE)
    train_dl = torch.utils.data.DataLoader(DummyDataset())
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model, train_dl)
