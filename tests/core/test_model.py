import pytest
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from flash import ClassificationTask, Task

# ======== Mock functions ========


def dummy_metric(y_hat, y):
    return torch.zeros_like(y)


class DummyDataset(torch.utils.data.Dataset):

    def __getitem__(self, index):
        return torch.rand(1, 28, 28), torch.randint(10, size=(1, )).item()

    def __len__(self):
        return 100


# ================================


def test_init_trainer(tmpdir):
    mlp = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10), nn.LogSoftmax())
    model = Task(mlp, F.nll_loss)
    train_dl = torch.utils.data.DataLoader(DummyDataset())

    assert model.trainer is None
    model.fit(train_dl, fast_dev_run=True, default_root_dir=tmpdir)
    assert model.trainer is not None
    model.trainer.foo_bar = "foo bar"  # this attribute should stay if we fit again
    model.fit(train_dl, fast_dev_run=True, default_root_dir=tmpdir)
    assert model.trainer.foo_bar == "foo bar"

    model.fit(train_dl, fast_dev_run=True, default_root_dir=tmpdir, min_steps=1)
    # since we added an extra arg to the trainer, we will have to use a new one,
    # so it won't have the foo_bar attribute
    assert not hasattr(model.trainer, "foo_bar")


@pytest.mark.parametrize("metrics", [None, pl.metrics.Accuracy(), {"accuracy": pl.metrics.Accuracy()}])
def test_init_train(tmpdir, metrics):
    mlp = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10), nn.LogSoftmax())
    train_dl = torch.utils.data.DataLoader(DummyDataset())
    val_dl = torch.utils.data.DataLoader(DummyDataset())
    model = ClassificationTask(mlp, F.nll_loss, metrics=metrics)
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, train_dl, val_dl)
    trainer.test(model, val_dl)
