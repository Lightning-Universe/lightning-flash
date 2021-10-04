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

import pytest
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from flash import Trainer
from flash.core.utilities.imports import _IMAGE_AVAILABLE
from flash.image.classification import ImageClassifier


class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, index: int) -> Any:
        return {"input": torch.rand(3, 64, 64), "target": torch.randint(10, size=(1,)).item()}

    def __len__(self) -> int:
        return 100


@pytest.mark.skipif(not _IMAGE_AVAILABLE, reason="image libraries aren't installed.")
@pytest.mark.parametrize(
    "strategy",
    [
        "no_freeze",
        "freeze",
        ("freeze_unfreeze", 1),
        ("unfreeze_milestones", ((5, 10), 5)),
    ],
)
def test_finetuning(tmpdir: str, strategy):
    train_dl = torch.utils.data.DataLoader(DummyDataset())
    val_dl = torch.utils.data.DataLoader(DummyDataset())
    task = ImageClassifier(10, backbone="resnet18")
    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.finetune(task, train_dl, val_dl, strategy=strategy)


@pytest.mark.skipif(not _IMAGE_AVAILABLE, reason="image libraries aren't installed.")
@pytest.mark.parametrize(
    "strategy",
    [
        None,
        "chocolate",
        (12, 1),
        ("chocolate", 1),
        ("freeze_unfreeze", "True"),
        ("unfreeze_milestones", "False"),
        ("unfreeze_milestones", (10, 10)),
        ("unfreeze_milestones", (10, (10, 10))),
        ("unfreeze_milestones", ((10, 10), "True")),
        ("unfreeze_milestones", ((3.14, 10), 10)),
        ("unfreeze_milestones", ((10, 3.14), 10)),
    ],
)
def test_finetuning_errors_and_exceptions(tmpdir: str, strategy):
    train_dl = torch.utils.data.DataLoader(DummyDataset())
    val_dl = torch.utils.data.DataLoader(DummyDataset())
    task = ImageClassifier(10, backbone="resnet18")
    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    with pytest.raises(MisconfigurationException):
        trainer.finetune(task, train_dl, val_dl, strategy=strategy)
