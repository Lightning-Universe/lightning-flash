import torch
from pytorch_lightning import Trainer

from flash.tabular import TabularClassifier

# ======== Mock functions ========


class DummyDataset(torch.utils.data.Dataset):

    def __init__(self, num_num=16, num_cat=16):
        super().__init__()
        self.num_num = num_num
        self.num_cat = num_cat

    def __getitem__(self, index):
        target = torch.randint(0, 10, size=(1, )).item()
        cat_vars = torch.randint(0, 10, size=(self.num_cat, ))
        num_vars = torch.rand(self.num_num)
        return (cat_vars, num_vars), target

    def __len__(self):
        return 100


# ==============================


def test_init_train(tmpdir):
    train_dl = torch.utils.data.DataLoader(DummyDataset(), batch_size=16)
    model = TabularClassifier(num_classes=10, num_features=16 + 16, embedding_sizes=16 * [(10, 32)])
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model, train_dl)


def test_init_train_no_num(tmpdir):
    train_dl = torch.utils.data.DataLoader(DummyDataset(num_num=0), batch_size=16)
    model = TabularClassifier(num_classes=10, num_features=16, embedding_sizes=16 * [(10, 32)])
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model, train_dl)


def test_init_train_no_cat(tmpdir):
    train_dl = torch.utils.data.DataLoader(DummyDataset(num_cat=0), batch_size=16)
    model = TabularClassifier(num_classes=10, num_features=16, embedding_sizes=[])
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model, train_dl)
