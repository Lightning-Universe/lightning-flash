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

import torch
from torch import nn
from torch.nn import functional as F

from flash import ClassificationTask, Trainer
from flash.core.finetuning import NoFreeze


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


class DummyClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
        self.head = nn.LogSoftmax()

    def forward(self, x):
        return self.head(self.backbone(x))


def test_task_fit(tmpdir: str):
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10), nn.LogSoftmax())
    train_dl = torch.utils.data.DataLoader(DummyDataset())
    val_dl = torch.utils.data.DataLoader(DummyDataset())
    task = ClassificationTask(model, F.nll_loss)
    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    result = trainer.fit(task, train_dl, val_dl)
    assert result


def test_task_finetune(tmpdir: str):
    model = DummyClassifier()
    train_dl = torch.utils.data.DataLoader(DummyDataset())
    val_dl = torch.utils.data.DataLoader(DummyDataset())
    task = ClassificationTask(model, F.nll_loss)
    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    result = trainer.finetune(task, train_dl, val_dl, strategy=NoFreeze())
    assert result
