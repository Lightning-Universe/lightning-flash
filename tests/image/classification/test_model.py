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
from torch import Tensor

from flash import Trainer
from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _IMAGE_AVAILABLE, _IMAGE_TESTING, _SERVE_TESTING
from flash.image import ImageClassifier
from tests.helpers.task_tester import TaskTester

# ======== Mock functions ========


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
    cli_command = "image_classification"
    is_testing = _IMAGE_TESTING
    is_available = _IMAGE_AVAILABLE

    marks = {
        "test_fit": [
            pytest.mark.parametrize(
                "task_kwargs",
                [
                    {"backbone": "resnet18"},
                    {"backbone": "vit_small_patch16_224"},
                    {"backbone": "resnet18", "head": "linear"},
                    {"backbone": "resnet18", "head": torch.nn.Linear(512, 2)},
                    {"backbone": "clip_resnet50"},
                ],
            )
        ],
        "test_val": [
            pytest.mark.parametrize(
                "task_kwargs",
                [
                    {"backbone": "resnet18"},
                    {"backbone": "vit_small_patch16_224"},
                    {"backbone": "resnet18", "head": "linear"},
                    {"backbone": "resnet18", "head": torch.nn.Linear(512, 2)},
                    {"backbone": "clip_resnet50"},
                ],
            )
        ],
        "test_test": [
            pytest.mark.parametrize(
                "task_kwargs",
                [
                    {"backbone": "resnet18"},
                    {"backbone": "vit_small_patch16_224"},
                    {"backbone": "resnet18", "head": "linear"},
                    {"backbone": "resnet18", "head": torch.nn.Linear(512, 2)},
                    {"backbone": "clip_resnet50"},
                ],
            )
        ],
        "test_cli": [pytest.mark.parametrize("extra_args", ([], ["from_movie_posters"]))],
    }

    # FIXME: jit script is failing for leaking `use_amp` which was removed in PL 1.8
    # @property
    # def example_forward_input(self):
    #     return torch.rand(1, 3, 32, 32)

    def check_forward_output(self, output: Any):
        assert isinstance(output, Tensor)
        assert output.shape == torch.Size([1, 2])

    @property
    def example_train_sample(self):
        return {DataKeys.INPUT: torch.rand(3, 224, 224), DataKeys.TARGET: 1}

    @property
    def example_val_sample(self):
        return self.example_train_sample

    @property
    def example_test_sample(self):
        return self.example_train_sample


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
