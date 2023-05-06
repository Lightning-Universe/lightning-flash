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
from typing import Any, Tuple, Union

import pytest
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning.core.lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from flash import Trainer
from flash.core.classification import ClassificationTask
from flash.core.utilities.imports import _TOPIC_CORE_AVAILABLE


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, predict: bool = False):
        self._predict = predict

    def __getitem__(self, index: int) -> Any:
        sample = torch.rand(1, 28, 28)
        if self._predict:
            return sample
        return sample, torch.randint(10, size=(1,)).item()

    def __len__(self) -> int:
        return 100


class DummyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
        self.head = nn.LogSoftmax()

    def forward(self, x):
        return self.head(self.backbone(x))


class NoFreeze(BaseFinetuning):
    def freeze_before_training(self, pl_module: LightningModule) -> None:
        pass

    def finetune_function(
        self,
        pl_module: LightningModule,
        epoch: int,
        optimizer: Optimizer,
        opt_idx: int,
    ) -> None:
        pass


@pytest.mark.skipif(not _TOPIC_CORE_AVAILABLE, reason="Not testing core.")
@pytest.mark.parametrize("callbacks, should_warn", [([], False), ([NoFreeze()], True)])
def test_trainer_fit(tmpdir, callbacks, should_warn):
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10), nn.LogSoftmax())
    train_dl = DataLoader(DummyDataset())
    val_dl = DataLoader(DummyDataset())
    task = ClassificationTask(model, loss_fn=F.nll_loss)
    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir, callbacks=callbacks)

    if should_warn:
        with pytest.warns(UserWarning, match="trainer is using a fine-tuning callback"):
            trainer.fit(task, train_dl, val_dl)
    else:
        trainer.fit(task, train_dl, val_dl)


@pytest.mark.skipif(not _TOPIC_CORE_AVAILABLE, reason="Not testing core.")
def test_trainer_finetune(tmpdir):
    model = DummyClassifier()
    train_dl = DataLoader(DummyDataset())
    val_dl = DataLoader(DummyDataset())
    task = ClassificationTask(model, loss_fn=F.nll_loss)
    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.finetune(task, train_dl, val_dl, strategy=NoFreeze())


@pytest.mark.skipif(not _TOPIC_CORE_AVAILABLE, reason="Not testing core.")
def test_resolve_callbacks_invalid_strategy(tmpdir):
    model = DummyClassifier()
    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    task = ClassificationTask(model, loss_fn=F.nll_loss)
    with pytest.raises(TypeError, match="should be a ``pytorch_lightning.callbacks.BaseFinetuning``"):
        trainer._resolve_callbacks(task, EarlyStopping("test"))


class MultiFinetuneClassificationTask(ClassificationTask):
    def configure_finetune_callback(
        self,
        strategy: Union[str, BaseFinetuning, Tuple[str, int], Tuple[str, Tuple[int, int]]] = "no_freeze",
        train_bn: bool = True,
    ):
        return [NoFreeze(), NoFreeze()]


@pytest.mark.skipif(not _TOPIC_CORE_AVAILABLE, reason="Not testing core.")
def test_resolve_callbacks_multi_error(tmpdir):
    model = DummyClassifier()
    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    task = MultiFinetuneClassificationTask(model, loss_fn=F.nll_loss)
    with pytest.raises(ValueError):
        trainer._resolve_callbacks(task, None)


@pytest.mark.skipif(not _TOPIC_CORE_AVAILABLE, reason="Not testing core.")
def test_resolve_callbacks_override_warning(tmpdir):
    model = DummyClassifier()
    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    task = ClassificationTask(model, loss_fn=F.nll_loss)
    with pytest.warns(UserWarning, match="The model contains a default finetune callback"):
        trainer._resolve_callbacks(task, strategy="no_freeze")


@pytest.mark.skipif(not _TOPIC_CORE_AVAILABLE, reason="Not testing core.")
def test_add_argparse_args():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args(["--gpus=1"])
    assert args.gpus == 1


@pytest.mark.skipif(not _TOPIC_CORE_AVAILABLE, reason="Not testing core.")
def test_from_argparse_args():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args(["--max_epochs=200"])
    trainer = Trainer.from_argparse_args(args)
    assert trainer.max_epochs == 200
    assert isinstance(trainer, Trainer)
