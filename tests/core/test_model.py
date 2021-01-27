from typing import Any

import pytest
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from flash import ClassificationTask

# ======== Mock functions ========


class DummyDataset(torch.utils.data.Dataset):

    def __init__(self, predict: bool = False):
        self._predict = predict

    def __getitem__(self, index: int) -> Any:
        sample = torch.rand(1, 28, 28)
        if self._predict:
            return sample
        else:
            return sample, torch.randint(10, size=(1, )).item()

    def __len__(self) -> int:
        return 100


# ================================


@pytest.mark.parametrize("metrics", [None, pl.metrics.Accuracy(), {"accuracy": pl.metrics.Accuracy()}])
def test_classificationtask_train(tmpdir: str, metrics: Any):
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10), nn.LogSoftmax())
    train_dl = torch.utils.data.DataLoader(DummyDataset())
    val_dl = torch.utils.data.DataLoader(DummyDataset())
    task = ClassificationTask(model, F.nll_loss, metrics=metrics)
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    result = trainer.fit(task, train_dl, val_dl)
    assert result
    result = trainer.test(task, val_dl)
    assert "test_nll_loss" in result[0]


def test_classificationtask_task_predict():
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    task = ClassificationTask(model)
    ds = DummyDataset()
    expected = list(range(10))
    # single item
    x0, _ = ds[0]
    pred0 = task.predict(x0)
    assert pred0[0] in expected
    # list
    x1, _ = ds[1]
    pred1 = task.predict([x0, x1])
    assert all(c in expected for c in pred1)
    assert pred0[0] == pred1[0]


def test_classificationtask_trainer_predict(tmpdir):
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    task = ClassificationTask(model)
    ds = DummyDataset(predict=True)
    batch_size = 3
    predict_dl = torch.utils.data.DataLoader(ds, batch_size=batch_size)
    trainer = pl.Trainer(default_root_dir=tmpdir)
    expected = list(range(10))
    predictions = trainer.predict(task, predict_dl)
    predictions = predictions[0]  # dataloader 0 predictions
    for pred in predictions[:-1]:
        # check batch sizes are correct
        assert len(pred) == batch_size
        assert all(c in expected for c in pred)
    # check size of last batch (not full)
    assert len(predictions[-1]) == len(ds) % batch_size
