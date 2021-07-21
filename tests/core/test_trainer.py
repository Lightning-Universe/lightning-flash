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
from argparse import ArgumentParser
from typing import Any

import pytest
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.nn import functional as F

from flash import Trainer
from flash.core.classification import ClassificationTask
from flash.core.finetuning import NoFreeze


class DummyDataset(torch.utils.data.Dataset):

    def __init__(self, predict: bool = False):
        self._predict = predict

    def __getitem__(self, index: int) -> Any:
        sample = torch.rand(1, 28, 28)
        if self._predict:
            return sample
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


@pytest.mark.parametrize("callbacks, should_warn", [([], False), ([NoFreeze()], True)])
def test_trainer_fit(tmpdir, callbacks, should_warn):
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10), nn.LogSoftmax())
    train_dl = torch.utils.data.DataLoader(DummyDataset())
    val_dl = torch.utils.data.DataLoader(DummyDataset())
    task = ClassificationTask(model, loss_fn=F.nll_loss)
    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir, callbacks=callbacks)

    if should_warn:
        with pytest.warns(UserWarning, match="trainer is using a fine-tuning callback"):
            trainer.fit(task, train_dl, val_dl)
    else:
        trainer.fit(task, train_dl, val_dl)


def test_trainer_finetune(tmpdir):
    model = DummyClassifier()
    train_dl = torch.utils.data.DataLoader(DummyDataset())
    val_dl = torch.utils.data.DataLoader(DummyDataset())
    task = ClassificationTask(model, loss_fn=F.nll_loss)
    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.finetune(task, train_dl, val_dl, strategy=NoFreeze())


def test_resolve_callbacks_invalid_strategy(tmpdir):
    model = DummyClassifier()
    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    task = ClassificationTask(model, loss_fn=F.nll_loss)
    with pytest.raises(MisconfigurationException, match="should be a ``pytorch_lightning.callbacks.BaseFinetuning``"):
        trainer._resolve_callbacks(task, EarlyStopping())


class MultiFinetuneClassificationTask(ClassificationTask):

    def configure_finetune_callback(self):
        return [NoFreeze(), NoFreeze()]


def test_resolve_callbacks_multi_error(tmpdir):
    model = DummyClassifier()
    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    task = MultiFinetuneClassificationTask(model, loss_fn=F.nll_loss)
    with pytest.raises(MisconfigurationException, match="should create a list with only 1 callback"):
        trainer._resolve_callbacks(task, None)


class FinetuneClassificationTask(ClassificationTask):

    def configure_finetune_callback(self):
        return [NoFreeze()]


def test_resolve_callbacks_override_warning(tmpdir):
    model = DummyClassifier()
    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    task = FinetuneClassificationTask(model, loss_fn=F.nll_loss)
    with pytest.warns(UserWarning, match="The model contains a default finetune callback"):
        trainer._resolve_callbacks(task, "test")


def test_add_argparse_args():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args(['--gpus=1'])
    assert args.gpus == 1


def test_from_argparse_args():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args(['--max_epochs=200'])
    trainer = Trainer.from_argparse_args(args)
    assert trainer.max_epochs == 200
    assert isinstance(trainer, Trainer)
