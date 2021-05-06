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
import pytest
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from flash import Trainer
from flash.core.classification import Probabilities
from flash.data.data_source import DefaultDataKeys
from flash.vision import ImageClassifier

# ======== Mock functions ========


class DummyDataset(torch.utils.data.Dataset):

    def __getitem__(self, index):
        return {
            DefaultDataKeys.INPUT: torch.rand(3, 224, 224),
            DefaultDataKeys.TARGET: torch.randint(10, size=(1, )).item(),
        }

    def __len__(self) -> int:
        return 100


class DummyMultiLabelDataset(torch.utils.data.Dataset):

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def __getitem__(self, index):
        return {
            DefaultDataKeys.INPUT: torch.rand(3, 224, 224),
            DefaultDataKeys.TARGET: torch.randint(0, 2, (self.num_classes, )),
        }

    def __len__(self) -> int:
        return 100


# ==============================


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
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.finetune(model, train_dl, strategy="freeze_unfreeze")


def test_non_existent_backbone():
    with pytest.raises(KeyError):
        ImageClassifier(2, "i am never going to implement this lol")


def test_freeze():
    model = ImageClassifier(2)
    model.freeze()
    for p in model.backbone.parameters():
        assert p.requires_grad is False


def test_unfreeze():
    model = ImageClassifier(2)
    model.unfreeze()
    for p in model.backbone.parameters():
        assert p.requires_grad is True


def test_multilabel(tmpdir):

    num_classes = 4
    ds = DummyMultiLabelDataset(num_classes)
    model = ImageClassifier(num_classes, multi_label=True, serializer=Probabilities(multi_label=True))
    train_dl = torch.utils.data.DataLoader(ds, batch_size=2)
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.finetune(model, train_dl, strategy="freeze_unfreeze")
    image, label = ds[0][DefaultDataKeys.INPUT], ds[0][DefaultDataKeys.TARGET]
    predictions = model.predict([{DefaultDataKeys.INPUT: image}])
    assert (torch.tensor(predictions) > 1).sum() == 0
    assert (torch.tensor(predictions) < 0).sum() == 0
    assert len(predictions[0]) == num_classes == len(label)
    assert len(torch.unique(label)) <= 2
