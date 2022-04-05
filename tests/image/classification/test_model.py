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
from typing import Any
from unittest import mock

import pytest
import torch

from flash import Trainer
from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _IMAGE_TESTING, _SERVE_TESTING
from flash.image import ImageClassifier
from tests.helpers.task_tester import TaskTester

# ======== Mock functions ========


class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        return {
            DataKeys.INPUT: torch.rand(3, 224, 224),
            DataKeys.TARGET: torch.randint(10, size=(1,)).item(),
        }

    def __len__(self) -> int:
        return 100


class DummyMultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def __getitem__(self, index):
        return {
            DataKeys.INPUT: torch.rand(3, 224, 224),
            DataKeys.TARGET: torch.randint(0, 2, (self.num_classes,)),
        }

    def __len__(self) -> int:
        return 100


# ==============================


class TestImageClassifier(TaskTester):

    task = ImageClassifier
    task_args = (2,)
    forward_input_shape = (3, 32, 32)
    cli_command = "image_classification"

    def check_forward_output(self, output: Any):
        assert isinstance(output, torch.Tensor)
        assert output.shape == torch.Size([1, 2])


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
@pytest.mark.parametrize(
    "backbone,metrics",
    [
        ("resnet18", None),
        ("resnet18", []),
        # "resnet34",
        # "resnet50",
        # "resnet101",
        # "resnet152",
    ],
)
def test_init_train(tmpdir, backbone, metrics):
    model = ImageClassifier(10, backbone=backbone, metrics=metrics)
    train_dl = torch.utils.data.DataLoader(DummyDataset())
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.finetune(model, train_dl, strategy="freeze")


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
@pytest.mark.parametrize("head", ["linear", torch.nn.Linear(512, 10)])
def test_init_train_head(tmpdir, head):
    model = ImageClassifier(10, backbone="resnet18", head=head, metrics=None)
    train_dl = torch.utils.data.DataLoader(DummyDataset())
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.finetune(model, train_dl, strategy="freeze")


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_non_existent_backbone():
    with pytest.raises(KeyError):
        ImageClassifier(2, backbone="i am never going to implement this lol")


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_freeze():
    model = ImageClassifier(2)
    model.freeze()
    for p in model.backbone.parameters():
        assert p.requires_grad is False


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_unfreeze():
    model = ImageClassifier(2)
    model.unfreeze()
    for p in model.backbone.parameters():
        assert p.requires_grad is True


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_multilabel(tmpdir):

    num_classes = 4
    ds = DummyMultiLabelDataset(num_classes)
    model = ImageClassifier(num_classes, multi_label=True)
    train_dl = torch.utils.data.DataLoader(ds, batch_size=2)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, limit_train_batches=5)
    trainer.finetune(model, train_dl, strategy=("freeze_unfreeze", 1))
    predictions = trainer.predict(model, train_dl, output="probabilities")[0]
    assert (torch.tensor(predictions) > 1).sum() == 0
    assert (torch.tensor(predictions) < 0).sum() == 0
    assert len(predictions[0]) == num_classes


@pytest.mark.skipif(not _SERVE_TESTING, reason="serve libraries aren't installed.")
@mock.patch("flash._IS_TESTING", True)
def test_serve():
    model = ImageClassifier(2)
    model.eval()
    model.serve()
