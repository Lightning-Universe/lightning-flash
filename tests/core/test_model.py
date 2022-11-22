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
from typing import Any, Tuple
from unittest import mock
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import Callback
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

import flash
from flash import Task
from flash.audio import SpeechRecognition
from flash.core.adapter import Adapter
from flash.core.classification import ClassificationTask
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.io.output_transform import OutputTransform
from flash.core.utilities.embedder import Embedder
from flash.core.utilities.imports import (
    _AUDIO_TESTING,
    _CORE_TESTING,
    _GRAPH_TESTING,
    _IMAGE_AVAILABLE,
    _IMAGE_TESTING,
    _PL_GREATER_EQUAL_1_8_0,
    _TABULAR_TESTING,
    _TEXT_TESTING,
    _TORCH_OPTIMIZER_AVAILABLE,
    _TRANSFORMERS_AVAILABLE,
)
from flash.graph import GraphClassifier, GraphEmbedder
from flash.image import ImageClassifier, SemanticSegmentation
from flash.tabular import TabularClassifier
from flash.text import SummarizationTask, TextClassifier, TranslationTask

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


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
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


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
def test_task_predict_raises():
    with pytest.raises(AttributeError, match="`flash.Task.predict` has been removed."):
        model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10), nn.Softmax())
        task = ClassificationTask(model, loss_fn=F.nll_loss)
        task.predict("args", kwarg="test")


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
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


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
def test_classification_task_trainer_predict(tmpdir):
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    task = ClassificationTask(model)
    ds = PredictDummyDataset(10)
    batch_size = 6
    predict_dl = task.process_predict_dataset(ds, input_transform=InputTransform(), batch_size=batch_size)
    trainer = pl.Trainer(default_root_dir=tmpdir)
    predictions = trainer.predict(task, predict_dl)
    assert len(list(chain.from_iterable(predictions))) == 10


@pytest.mark.parametrize(
    ["cls", "filename"],
    [
        pytest.param(
            ImageClassifier,
            "0.7.0/image_classification_model.pt",
            marks=pytest.mark.skipif(
                not _IMAGE_TESTING,
                reason="image packages aren't installed",
            ),
        ),
        pytest.param(
            SemanticSegmentation,
            "0.9.0/semantic_segmentation_model.pt",
            marks=pytest.mark.skipif(
                not _IMAGE_TESTING,
                reason="image packages aren't installed",
            ),
        ),
        pytest.param(
            SpeechRecognition,
            "0.7.0/speech_recognition_model.pt",
            marks=pytest.mark.skipif(
                not _AUDIO_TESTING,
                reason="audio packages aren't installed",
            ),
        ),
        pytest.param(
            TabularClassifier,
            "0.7.0/tabular_classification_model.pt",
            marks=pytest.mark.skipif(
                not _TABULAR_TESTING,
                reason="tabular packages aren't installed",
            ),
        ),
        pytest.param(
            TextClassifier,
            "0.9.0/text_classification_model.pt",
            marks=pytest.mark.skipif(
                not _TEXT_TESTING,
                reason="text packages aren't installed",
            ),
        ),
        pytest.param(
            SummarizationTask,
            "0.7.0/summarization_model_xsum.pt",
            marks=pytest.mark.skipif(
                not _TEXT_TESTING,
                reason="text packages aren't installed",
            ),
        ),
        pytest.param(
            TranslationTask,
            "0.7.0/translation_model_en_ro.pt",
            marks=pytest.mark.skipif(
                not _TEXT_TESTING,
                reason="text packages aren't installed",
            ),
        ),
        pytest.param(
            GraphClassifier,
            "0.7.0/graph_classification_model.pt",
            marks=pytest.mark.skipif(
                not _GRAPH_TESTING,
                reason="graph packages aren't installed",
            ),
        ),
        pytest.param(
            GraphEmbedder,
            "0.7.0/graph_classification_model.pt",
            marks=pytest.mark.skipif(
                not _GRAPH_TESTING,
                reason="graph packages aren't installed",
            ),
        ),
    ],
)
def test_model_download(tmpdir, cls, filename):
    url = "https://flash-weights.s3.amazonaws.com/"
    with tmpdir.as_cwd():
        task = cls.load_from_checkpoint(url + filename)
        assert isinstance(task, cls)


class DummyTask(Task):
    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 30),
            nn.Linear(30, 40),
        )

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        return self.backbone(batch)


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
def test_as_embedder():
    layer_number = 1
    embedder = DummyTask().as_embedder(f"backbone.{layer_number}")

    assert isinstance(embedder, Embedder)
    assert embedder.predict_step(torch.rand(10, 10), 0, 0).size(1) == embedder.model.backbone[layer_number].out_features


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
def test_available_layers():
    task = DummyTask()
    assert task.available_layers() == ["output", "", "backbone", "backbone.0", "backbone.1", "backbone.2"]


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_available_backbones():
    backbones = ImageClassifier.available_backbones()
    assert "resnet152" in backbones

    class Foo(ImageClassifier):
        backbones = None

    assert Foo.available_backbones() is None


@pytest.mark.skipif(_IMAGE_AVAILABLE, reason="image libraries are installed.")
def test_available_backbones_raises():
    with pytest.raises(ModuleNotFoundError, match="Required dependencies not available."):
        _ = ImageClassifier.available_backbones()


@ClassificationTask.lr_schedulers_registry
def custom_steplr_configuration_return_as_instance(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)


@ClassificationTask.lr_schedulers_registry
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


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
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


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
def test_optimizer_learning_rate():
    mock_optimizer = MagicMock()
    Task.optimizers_registry(mock_optimizer, "test")

    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10), nn.LogSoftmax())

    ClassificationTask(model, optimizer="test").configure_optimizers()
    mock_optimizer.assert_called_once_with(mock.ANY)

    mock_optimizer.reset_mock()

    ClassificationTask(model, optimizer="test", learning_rate=10).configure_optimizers()
    mock_optimizer.assert_called_once_with(mock.ANY, lr=10)

    mock_optimizer.reset_mock()

    with pytest.raises(TypeError, match="The `learning_rate` argument is required"):
        ClassificationTask(model, optimizer="sgd").configure_optimizers()


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
@pytest.mark.parametrize("use_datamodule", [False, True])
@pytest.mark.parametrize("limit", [None, 5, 0.1])
def test_external_schedulers_provider_hf_transformers(tmpdir, optim, sched, use_datamodule, limit):
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10), nn.LogSoftmax())
    task = ClassificationTask(model, optimizer=deepcopy(optim), lr_scheduler=deepcopy(sched), loss_fn=F.nll_loss)

    if limit is not None:
        batch_count = limit if isinstance(limit, int) else int(limit * 10)
        trainer = flash.Trainer(max_epochs=1, limit_train_batches=limit)
    else:
        batch_count = 10
        trainer = flash.Trainer(max_epochs=1)

    ds = DummyDataset(num_samples=10)

    if use_datamodule:

        class TestDataModule(LightningDataModule):
            def train_dataloader(self):
                return DataLoader(ds)

        trainer.fit(task, datamodule=TestDataModule())
    else:
        trainer.fit(task, train_dataloader=DataLoader(ds))

    assert task.get_num_training_steps() == batch_count
    assert isinstance(trainer.optimizers[0], torch.optim.Adadelta)
    if _PL_GREATER_EQUAL_1_8_0:
        assert isinstance(trainer.lr_scheduler_configs[0].scheduler, torch.optim.lr_scheduler.LambdaLR)
    else:
        assert isinstance(trainer.lr_schedulers[0]["scheduler"], torch.optim.lr_scheduler.LambdaLR)


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
def test_manual_optimization(tmpdir):
    class ManualOptimizationTask(Task):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.automatic_optimization = False

        def training_step(self, batch: Any, batch_idx: int) -> Any:
            optimizers = self.optimizers()
            assert isinstance(optimizers, torch.optim.Optimizer)
            optimizers.zero_grad()

            output = self.step(batch, batch_idx, self.train_metrics)
            self.manual_backward(output["loss"])

            optimizers.step()

            lr_schedulers = self.lr_schedulers()
            assert isinstance(lr_schedulers, torch.optim.lr_scheduler._LRScheduler)
            lr_schedulers.step()

    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10), nn.Softmax())
    train_dl = DataLoader(DummyDataset())
    val_dl = DataLoader(DummyDataset())
    task = ManualOptimizationTask(model, loss_fn=F.nll_loss, lr_scheduler=("steplr", {"step_size": 1}))

    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(task, train_dl, val_dl)


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
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

    with pytest.raises(TypeError):
        task = ClassificationTask(model, optimizer="Adam", lr_scheduler=["not", "a", "valid", "type"])
        task.configure_optimizers()

    with pytest.raises(TypeError):
        task = ClassificationTask(
            model, optimizer="Adam", lr_scheduler=(["not", "a", "valid", "type"], {"random_kwarg": 10})
        )
        task.configure_optimizers()


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
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
    trainer.test(task, DataLoader(test_dataset))


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
def test_loss_fn_buffer():
    weight = torch.rand(10)
    model = Task(loss_fn=nn.CrossEntropyLoss(weight=weight))
    state_dict = model.state_dict()

    assert len(state_dict) == 1
    assert torch.allclose(state_dict["loss_fn.crossentropyloss.weight"], weight)
