import pytest
import torch
import pytorch_lightning as pl

from pl_flash.vision import ImageClassifier

from unittest.mock import patch
from imp import reload

###### Mock functions #####
class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        return torch.rand(3, 224, 224), torch.randint(10, size=(1,)).item()

    def __len__(self):
        return 100


##########################


@pytest.mark.parametrize(
    "backbone",
    [
        "resnet18",
        # "resnet34",
        # "resnet50",
        # "resnet101",
        # "resnet152",
    ],
)
def test_init_train(tmpdir, backbone):
    model = ImageClassifier(10, backbone=backbone)
    train_dl = torch.utils.data.DataLoader(DummyDataset())
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, train_dl)


def test_non_existent_backbone():
    with pytest.raises(NotImplementedError):
        model = ImageClassifier(2, "i am never going to implement this lol")


def test_freeze():
    model = ImageClassifier(2)
    model.freeze()
    for p in model.backbone.parameters():
        assert p.requires_grad == False


def test_unfreeze():
    model = ImageClassifier(2)
    model.unfreeze()
    for p in model.backbone.parameters():
        assert p.requires_grad == True
