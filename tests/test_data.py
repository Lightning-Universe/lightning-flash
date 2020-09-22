import torch
from unittest.mock import patch

from pl_flash import DataModule


###### Mock functions #####
def dummy_metric(y_hat, y):
    return torch.zeros_like(y)


class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        return torch.rand(1, 28, 28), torch.randint(10, size=(1,)).item()

    def __len__(self):
        return 10


##########################


def test_init():
    train_ds, val_ds, test_ds = DummyDataset(), DummyDataset(), DummyDataset()
    dm = DataModule(train_ds)
    dm = DataModule(train_ds, val_ds)
    dm = DataModule(train_ds, val_ds, test_ds)


def test_dataloaders():
    train_ds, val_ds, test_ds = DummyDataset(), DummyDataset(), DummyDataset()
    dm = DataModule(train_ds, val_ds, test_ds, num_workers=0)
    for dl in [
        dm.train_dataloader(),
        dm.val_dataloader(),
        dm.test_dataloader(),
    ]:
        x, y = next(iter(dl))
        assert x.shape == (1, 1, 28, 28)


def test_cpu_count_none():
    train_ds = DummyDataset()

    with patch("os.cpu_count", return_value=None):
        dm = DataModule(train_ds, num_workers=None)
    assert dm.num_workers == 0
