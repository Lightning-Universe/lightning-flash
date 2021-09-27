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
import math
from itertools import chain
from numbers import Number
from pathlib import Path
from typing import Any, Tuple
from unittest import mock

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

import flash
from flash.core.adapter import Adapter
from flash.core.classification import ClassificationTask
from flash.core.data.process import DefaultPreprocess, Postprocess
from flash.core.utilities.imports import _TABULAR_AVAILABLE, _TEXT_AVAILABLE, Image
from flash.image import ImageClassificationData, ImageClassifier
from tests.helpers.utils import _IMAGE_TESTING, _TABULAR_TESTING

if _TABULAR_AVAILABLE:
    from flash.tabular import TabularClassifier
else:
    TabularClassifier = None


# ======== Mock functions ========


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples: int = 9):
        self.num_samples = num_samples

    def __getitem__(self, index: int) -> Tuple[Tensor, Number]:
        return torch.rand(1, 28, 28), torch.randint(10, size=(1,)).item()

    def __len__(self) -> int:
        return self.num_samples


class PredictDummyDataset(DummyDataset):
    def __init__(self, num_samples: int):
        super().__init__(num_samples)

    def __getitem__(self, index: int) -> Tensor:
        return torch.rand(1, 28, 28)


class DummyPostprocess(Postprocess):

    pass


class FixedDataset(torch.utils.data.Dataset):
    def __init__(self, targets):
        super().__init__()

        self.targets = targets

    def __getitem__(self, index: int) -> Tuple[Tensor, Number]:
        return torch.rand(1), self.targets[index]

    def __len__(self) -> int:
        return len(self.targets)


class OnesModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer = nn.Linear(1, 2)
        self.register_buffer("zeros", torch.zeros(2))
        self.register_buffer("zero_one", torch.tensor([0.0, 1.0]))

    def forward(self, x):
        x = self.layer(x)
        return x * self.zeros + self.zero_one


class Parent(ClassificationTask):
    def __init__(self, child):
        super().__init__()

        self.child = child

    def training_step(self, batch, batch_idx):
        return self.child.training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.child.validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.child.test_step(batch, batch_idx)

    def forward(self, x):
        return self.child(x)


class GrandParent(Parent):
    def __init__(self, child):
        super().__init__(Parent(child))


class BasicAdapter(Adapter):
    def __init__(self, child):
        super().__init__()

        self.child = child

    def training_step(self, batch, batch_idx):
        return self.child.training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.child.validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.child.test_step(batch, batch_idx)

    def forward(self, x):
        return self.child(x)


class AdapterParent(Parent):
    def __init__(self, child):
        super().__init__(BasicAdapter(child))


# ================================


@pytest.mark.parametrize("metrics", [None, pl.metrics.Accuracy(), {"accuracy": pl.metrics.Accuracy()}])
def test_classificationtask_train(tmpdir: str, metrics: Any):
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10), nn.Softmax())
    train_dl = torch.utils.data.DataLoader(DummyDataset())
    val_dl = torch.utils.data.DataLoader(DummyDataset())
    task = ClassificationTask(model, loss_fn=F.nll_loss, metrics=metrics)
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    result = trainer.fit(task, train_dl, val_dl)
    result = trainer.test(task, val_dl)
    assert "test_nll_loss" in result[0]


@pytest.mark.parametrize("task", [Parent, GrandParent, AdapterParent])
def test_nested_tasks(tmpdir, task):
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10), nn.Softmax())
    train_dl = torch.utils.data.DataLoader(DummyDataset())
    val_dl = torch.utils.data.DataLoader(DummyDataset())
    child_task = ClassificationTask(model, loss_fn=F.nll_loss)

    parent_task = task(child_task)

    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(parent_task, train_dl, val_dl)
    result = trainer.test(parent_task, val_dl)
    assert "test_nll_loss" in result[0]


def test_classificationtask_task_predict():
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10), nn.Softmax())
    task = ClassificationTask(model, preprocess=DefaultPreprocess())
    ds = DummyDataset()
    expected = list(range(10))
    # single item
    x0, _ = ds[0]
    pred0 = task.predict(x0)
    assert pred0[0] in expected
    # list
    x1, _ = ds[1]
    pred1 = task.predict([x0, x1])
    assert all(c in expected for c in pred1)
    assert pred0[0] == pred1[0]


@mock.patch("flash._IS_TESTING", True)
@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_classification_task_predict_folder_path(tmpdir):
    train_dir = Path(tmpdir / "train")
    train_dir.mkdir()

    def _rand_image():
        return Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype="uint8"))

    _rand_image().save(train_dir / "1.png")
    _rand_image().save(train_dir / "2.png")

    datamodule = ImageClassificationData.from_folders(predict_folder=train_dir)

    task = ImageClassifier(num_classes=10)
    predictions = task.predict(str(train_dir), data_pipeline=datamodule.data_pipeline)
    assert len(predictions) == 2


def test_classification_task_trainer_predict(tmpdir):
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    task = ClassificationTask(model)
    ds = PredictDummyDataset(10)
    batch_size = 6
    predict_dl = task.process_predict_dataset(ds, batch_size=batch_size)
    trainer = pl.Trainer(default_root_dir=tmpdir)
    predictions = trainer.predict(task, predict_dl)
    assert len(list(chain.from_iterable(predictions))) == 10


def test_task_datapipeline_save(tmpdir):
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10), nn.Softmax())
    train_dl = torch.utils.data.DataLoader(DummyDataset())
    task = ClassificationTask(model, loss_fn=F.nll_loss, postprocess=DummyPostprocess())

    # to check later
    task.postprocess.test = True

    # generate a checkpoint
    trainer = pl.Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=1,
        max_epochs=1,
        progress_bar_refresh_rate=0,
        weights_summary=None,
        logger=False,
    )
    trainer.fit(task, train_dl)
    path = str(tmpdir / "model.ckpt")
    trainer.save_checkpoint(path)

    # load from file
    task = ClassificationTask.load_from_checkpoint(path, model=model)
    assert task.postprocess.test


@pytest.mark.parametrize(
    ["cls", "filename"],
    [
        # needs to be updated.
        # pytest.param(
        #    ImageClassifier,
        #    "image_classification_model.pt",
        #    marks=pytest.mark.skipif(
        #        not _IMAGE_TESTING,
        #        reason="image packages aren't installed",
        #    ),
        # ),
        pytest.param(
            TabularClassifier,
            "tabular_classification_model.pt",
            marks=pytest.mark.skipif(
                not _TABULAR_TESTING,
                reason="tabular packages aren't installed",
            ),
        ),
    ],
)
def test_model_download(tmpdir, cls, filename):
    url = "https://flash-weights.s3.amazonaws.com/"
    with tmpdir.as_cwd():
        task = cls.load_from_checkpoint(url + filename)
        assert isinstance(task, cls)


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_available_backbones():
    backbones = ImageClassifier.available_backbones()
    assert "resnet152" in backbones

    class Foo(ImageClassifier):
        backbones = None

    assert Foo.available_backbones() == {}


def test_optimization(tmpdir):

    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10), nn.LogSoftmax())
    optim = torch.optim.Adam(model.parameters())
    task = ClassificationTask(model, optimizer=optim, scheduler=None)

    optimizer = task.configure_optimizers()
    assert optimizer == optim

    task = ClassificationTask(model, optimizer=torch.optim.Adadelta, optimizer_kwargs={"eps": 0.5}, scheduler=None)
    optimizer = task.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adadelta)
    assert optimizer.defaults["eps"] == 0.5

    task = ClassificationTask(
        model,
        optimizer=torch.optim.Adadelta,
        scheduler=torch.optim.lr_scheduler.StepLR,
        scheduler_kwargs={"step_size": 1},
    )
    optimizer, scheduler = task.configure_optimizers()
    assert isinstance(optimizer[0], torch.optim.Adadelta)
    assert isinstance(scheduler[0], torch.optim.lr_scheduler.StepLR)

    optim = torch.optim.Adadelta(model.parameters())
    task = ClassificationTask(model, optimizer=optim, scheduler=torch.optim.lr_scheduler.StepLR(optim, step_size=1))
    optimizer, scheduler = task.configure_optimizers()
    assert isinstance(optimizer[0], torch.optim.Adadelta)
    assert isinstance(scheduler[0], torch.optim.lr_scheduler.StepLR)

    if _TEXT_AVAILABLE:
        from transformers.optimization import get_linear_schedule_with_warmup

        assert isinstance(task.available_schedulers(), list)

        optim = torch.optim.Adadelta(model.parameters())
        with pytest.raises(MisconfigurationException, match="The LightningModule isn't attached to the trainer yet."):
            task = ClassificationTask(model, optimizer=optim, scheduler="linear_schedule_with_warmup")
            optimizer, scheduler = task.configure_optimizers()

        task = ClassificationTask(
            model,
            optimizer=optim,
            scheduler="linear_schedule_with_warmup",
            scheduler_kwargs={"num_warmup_steps": 0.1},
            loss_fn=F.nll_loss,
        )
        trainer = flash.Trainer(max_epochs=1, limit_train_batches=2, gpus=torch.cuda.device_count())
        ds = DummyDataset()
        trainer.fit(task, train_dataloader=DataLoader(ds))
        optimizer, scheduler = task.configure_optimizers()
        assert isinstance(optimizer[0], torch.optim.Adadelta)
        assert isinstance(scheduler[0], torch.optim.lr_scheduler.LambdaLR)
        expected = get_linear_schedule_with_warmup.__name__
        assert scheduler[0].lr_lambdas[0].__qualname__.split(".")[0] == expected


def test_classification_task_metrics():
    train_dataset = FixedDataset([0, 1])
    val_dataset = FixedDataset([1, 1])

    model = OnesModel()

    class CheckAccuracy(Callback):
        def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
            assert math.isclose(trainer.callback_metrics["train_accuracy_epoch"], 0.5)

    task = ClassificationTask(model)
    trainer = flash.Trainer(max_epochs=1, callbacks=CheckAccuracy(), gpus=torch.cuda.device_count())
    trainer.fit(task, train_dataloader=DataLoader(train_dataset), val_dataloaders=DataLoader(val_dataset))
