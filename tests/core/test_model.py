import pytest
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from flash import ClassificationTask

# ======== Mock functions ========


class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        return torch.rand(1, 28, 28), torch.randint(10, size=(1, )).item()

    def __len__(self):
        return 100


# ================================


@pytest.mark.parametrize("metrics", [None, pl.metrics.Accuracy(), {"accuracy": pl.metrics.Accuracy()}])
def test_classificationtask_train(tmpdir, metrics):
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10), nn.LogSoftmax())
    train_dl = torch.utils.data.DataLoader(DummyDataset())
    val_dl = torch.utils.data.DataLoader(DummyDataset())
    task = ClassificationTask(model, F.nll_loss, metrics=metrics)
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    result = trainer.fit(task, train_dl, val_dl)
    assert result
    result = trainer.test(task, val_dl)
    assert "test_nll_loss" in result[0]
