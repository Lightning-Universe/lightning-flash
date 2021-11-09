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
import functools
import math
from copy import deepcopy
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
from torchmetrics import Accuracy

import flash
from flash.audio import SpeechRecognition
from flash.core.adapter import Adapter
from flash.core.classification import ClassificationTask
from flash.core.data.io.output_transform import OutputTransform
from flash.core.data.process import DefaultPreprocess
from flash.core.utilities.imports import _TORCH_OPTIMIZER_AVAILABLE, _TRANSFORMERS_AVAILABLE, Image
from flash.image import ImageClassificationData, ImageClassifier, SemanticSegmentation
from flash.tabular import TabularClassifier
from flash.text import SummarizationTask, TextClassifier, TranslationTask
from tests.helpers.utils import _AUDIO_TESTING, _IMAGE_TESTING, _TABULAR_TESTING, _TEXT_TESTING

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


class DummyOutputTransform(OutputTransform):

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


@pytest.mark.parametrize("metrics", [None, Accuracy(), {"accuracy": Accuracy()}])
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
    task = ClassificationTask(model, loss_fn=F.nll_loss, output_transform=DummyOutputTransform())

    # to check later
    task.output_transform.test = True

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
    assert task.output_transform.test


@pytest.mark.parametrize(
    ["cls", "filename"],
    [
        pytest.param(
            ImageClassifier,
            "0.6.0/image_classification_model.pt",
            marks=pytest.mark.skipif(
                not _IMAGE_TESTING,
                reason="image packages aren't installed",
            ),
        ),
        pytest.param(
            SemanticSegmentation,
            "0.6.0/semantic_segmentation_model.pt",
            marks=pytest.mark.skipif(
                not _IMAGE_TESTING,
                reason="image packages aren't installed",
            ),
        ),
        pytest.param(
            SpeechRecognition,
            "0.6.0/speech_recognition_model.pt",
            marks=pytest.mark.skipif(
                not _AUDIO_TESTING,
                reason="audio packages aren't installed",
            ),
        ),
        pytest.param(
            TabularClassifier,
            "0.6.0/tabular_classification_model.pt",
            marks=pytest.mark.skipif(
                not _TABULAR_TESTING,
                reason="tabular packages aren't installed",
            ),
        ),
        pytest.param(
            TextClassifier,
            "0.6.0/text_classification_model.pt",
            marks=pytest.mark.skipif(
                not _TEXT_TESTING,
                reason="text packages aren't installed",
            ),
        ),
        pytest.param(
            SummarizationTask,
            "0.6.0/summarization_model_xsum.pt",
            marks=pytest.mark.skipif(
                not _TEXT_TESTING,
                reason="text packages aren't installed",
            ),
        ),
        pytest.param(
            TranslationTask,
            "0.6.0/translation_model_en_ro.pt",
            marks=pytest.mark.skipif(
                not _TEXT_TESTING,
                reason="text packages aren't installed",
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


@ClassificationTask.lr_schedulers
def custom_steplr_configuration_return_as_instance(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)


@ClassificationTask.lr_schedulers
def custom_steplr_configuration_return_as_dict(optimizer):
    return {
        "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=10),
        "name": "A_Really_Cool_Name",
        "interval": "step",
        "frequency": 1,
        "reduce_on_plateau": False,
        "monitor": None,
        "strict": True,
        "opt_idx": None,
    }


@pytest.mark.parametrize(
    "optim", ["Adadelta", functools.partial(torch.optim.Adadelta, eps=0.5), ("Adadelta", {"eps": 0.5})]
)
@pytest.mark.parametrize(
    "sched, interval",
    [
        (None, "epoch"),
        ("custom_steplr_configuration_return_as_instance", "epoch"),
        ("custom_steplr_configuration_return_as_dict", "step"),
        (functools.partial(torch.optim.lr_scheduler.StepLR, step_size=10), "epoch"),
        (("StepLR", {"step_size": 10}), "step"),
        (("StepLR", {"step_size": 10}, {"interval": "epoch"}), "epoch"),
    ],
)
def test_optimizers_and_schedulers(tmpdir, optim, sched, interval):

    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10), nn.LogSoftmax())
    task = ClassificationTask(model, optimizer=optim, lr_scheduler=sched)
    train_dl = torch.utils.data.DataLoader(DummyDataset())

    if sched is None:
        optimizer = task.configure_optimizers()
        assert isinstance(optimizer, torch.optim.Adadelta)
    else:
        optimizer, scheduler = task.configure_optimizers()
        assert isinstance(optimizer[0], torch.optim.Adadelta)

        scheduler = scheduler[0]
        assert isinstance(scheduler["scheduler"], torch.optim.lr_scheduler.StepLR)
        assert scheduler["interval"] == interval

    # generate a checkpoint
    trainer = flash.Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=10,
        max_epochs=1,
    )
    trainer.fit(task, train_dl)


@pytest.mark.skipif(not _TORCH_OPTIMIZER_AVAILABLE, reason="torch_optimizer isn't installed.")
@pytest.mark.parametrize("optim", ["Yogi"])
def test_external_optimizers_torch_optimizer(tmpdir, optim):

    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10), nn.LogSoftmax())
    task = ClassificationTask(model, optimizer=optim, lr_scheduler=None, loss_fn=F.nll_loss)
    trainer = flash.Trainer(max_epochs=1, limit_train_batches=2, gpus=torch.cuda.device_count())
    ds = DummyDataset()
    trainer.fit(task, train_dataloader=DataLoader(ds))

    from torch_optimizer import Yogi

    optimizer = task.configure_optimizers()
    assert isinstance(optimizer, Yogi)


@pytest.mark.skipif(not _TRANSFORMERS_AVAILABLE, reason="transformers library isn't installed.")
@pytest.mark.parametrize("optim", ["Adadelta", functools.partial(torch.optim.Adadelta, eps=0.5)])
@pytest.mark.parametrize(
    "sched",
    [
        "constant_schedule",
        ("cosine_schedule_with_warmup", {"num_warmup_steps": 0.1}),
        ("cosine_with_hard_restarts_schedule_with_warmup", {"num_warmup_steps": 0.1, "num_cycles": 3}),
    ],
)
def test_external_schedulers_provider_hf_transformers(tmpdir, optim, sched):
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10), nn.LogSoftmax())
    task = ClassificationTask(model, optimizer=deepcopy(optim), lr_scheduler=deepcopy(sched), loss_fn=F.nll_loss)
    trainer = flash.Trainer(max_epochs=1, limit_train_batches=10, gpus=torch.cuda.device_count())
    ds = DummyDataset()
    trainer.fit(task, train_dataloader=DataLoader(ds))

    assert isinstance(trainer.optimizers[0], torch.optim.Adadelta)
    assert isinstance(trainer.lr_schedulers[0]["scheduler"], torch.optim.lr_scheduler.LambdaLR)


def test_errors_and_exceptions_optimizers_and_schedulers():
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10), nn.LogSoftmax())

    with pytest.raises(TypeError):
        task = ClassificationTask(model, optimizer=[1, 2, 3, 4], lr_scheduler=None)
        task.configure_optimizers()

    with pytest.raises(KeyError):
        task = ClassificationTask(model, optimizer="not_a_valid_key", lr_scheduler=None)
        task.configure_optimizers()

    with pytest.raises(TypeError):
        task = ClassificationTask(
            model, optimizer=(["not", "a", "valid", "type"], {"random_kwarg": 10}), lr_scheduler=None
        )
        task.configure_optimizers()

    with pytest.raises(TypeError):
        task = ClassificationTask(model, optimizer=("Adam", ["non", "dict", "type"]), lr_scheduler=None)
        task.configure_optimizers()

    with pytest.raises(KeyError):
        task = ClassificationTask(model, optimizer="Adam", lr_scheduler="not_a_valid_key")
        task.configure_optimizers()

    @ClassificationTask.lr_schedulers
    def i_will_create_a_misconfiguration_exception(optimizer):
        return "Done. Created."

    with pytest.raises(MisconfigurationException):
        task = ClassificationTask(model, optimizer="Adam", lr_scheduler="i_will_create_a_misconfiguration_exception")
        task.configure_optimizers()

    with pytest.raises(MisconfigurationException):
        task = ClassificationTask(model, optimizer="Adam", lr_scheduler=i_will_create_a_misconfiguration_exception)
        task.configure_optimizers()

    with pytest.raises(TypeError):
        task = ClassificationTask(model, optimizer="Adam", lr_scheduler=["not", "a", "valid", "type"])
        task.configure_optimizers()

    with pytest.raises(TypeError):
        task = ClassificationTask(
            model, optimizer="Adam", lr_scheduler=(["not", "a", "valid", "type"], {"random_kwarg": 10})
        )
        task.configure_optimizers()

    pass


def test_classification_task_metrics():
    train_dataset = FixedDataset([0, 1])
    val_dataset = FixedDataset([1, 1])
    test_dataset = FixedDataset([0, 0])

    model = OnesModel()

    class CheckAccuracy(Callback):
        def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
            assert math.isclose(trainer.callback_metrics["train_accuracy_epoch"], 0.5)

        def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
            assert math.isclose(trainer.callback_metrics["val_accuracy"], 1.0)

        def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
            assert math.isclose(trainer.callback_metrics["test_accuracy"], 0.0)

    task = ClassificationTask(model)
    trainer = flash.Trainer(max_epochs=1, callbacks=CheckAccuracy(), gpus=torch.cuda.device_count())
    trainer.fit(task, train_dataloader=DataLoader(train_dataset), val_dataloaders=DataLoader(val_dataset))
    trainer.test(task, dataloaders=DataLoader(test_dataset))
