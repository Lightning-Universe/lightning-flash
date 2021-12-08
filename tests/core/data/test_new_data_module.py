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
from dataclasses import dataclass
from typing import Callable, Dict

import torch
from pytorch_lightning import seed_everything
from torch.utils.data import Dataset

from flash import Task, Trainer
from flash.core.data.input_transform import InputTransform
from flash.core.data.io.input_base import Input
from flash.core.data.new_data_module import DataModule, DatasetInput
from flash.core.utilities.stages import RunningStage


def test_data_module():
    seed_everything(42)

    def train_fn(data):
        return data - 100

    def val_fn(data):
        return data + 100

    def test_fn(data):
        return data - 1000

    def predict_fn(data):
        return data + 1000

    @dataclass
    class TestTransform(InputTransform):
        def per_sample_transform(self):
            def fn(x):
                return x

            return fn

        def per_batch_transform_on_device(self) -> Callable:
            if self.training:
                return train_fn
            elif self.validating:
                return val_fn
            elif self.testing:
                return test_fn
            elif self.predicting:
                return predict_fn

    train_dataset = Input(RunningStage.TRAINING, range(10), transform=TestTransform)
    assert train_dataset.transform._running_stage == RunningStage.TRAINING
    assert train_dataset.running_stage == RunningStage.TRAINING

    transform = TestTransform(RunningStage.VALIDATING)
    assert transform._running_stage == RunningStage.VALIDATING
    val_dataset = Input(RunningStage.VALIDATING, range(10), transform=transform)
    assert val_dataset.running_stage == RunningStage.VALIDATING

    transform = TestTransform(RunningStage.TESTING)
    assert transform._running_stage == RunningStage.TESTING
    test_dataset = Input(RunningStage.TESTING, range(10), transform=transform)
    assert test_dataset.running_stage == RunningStage.TESTING

    transform = TestTransform(RunningStage.PREDICTING)
    assert transform._running_stage == RunningStage.PREDICTING
    predict_dataset = Input(RunningStage.PREDICTING, range(10), transform=transform)
    assert predict_dataset.running_stage == RunningStage.PREDICTING

    dm = DataModule(
        train_input=train_dataset,
        val_input=val_dataset,
        test_input=test_dataset,
        predict_input=predict_dataset,
        batch_size=2,
    )
    assert len(dm.train_dataloader()) == 5
    batch = next(iter(dm.train_dataloader()))
    assert batch.shape == torch.Size([2])
    assert batch.min() >= 0 and batch.max() < 10

    assert len(dm.val_dataloader()) == 5
    batch = next(iter(dm.val_dataloader()))
    assert batch.shape == torch.Size([2])
    assert batch.min() >= 0 and batch.max() < 10

    class TestModel(Task):
        def training_step(self, batch, batch_idx):
            assert sum(batch < 0) == 2

        def validation_step(self, batch, batch_idx):
            assert sum(batch > 0) == 2

        def test_step(self, batch, batch_idx):
            assert sum(batch < 500) == 2

        def predict_step(self, batch, batch_idx):
            assert sum(batch > 500) == 2
            assert torch.equal(batch, torch.tensor([1000, 1001]))

        def on_train_dataloader(self) -> None:
            pass

        def on_val_dataloader(self) -> None:
            pass

        def on_test_dataloader(self, *_) -> None:
            pass

        def on_predict_dataloader(self) -> None:
            pass

        def on_predict_end(self) -> None:
            pass

        def on_fit_end(self) -> None:
            pass

    model = TestModel(torch.nn.Linear(1, 1))
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)
    trainer.predict(model, dm)

    input = Input(RunningStage.TRAINING, transform=TestTransform)
    dm = DataModule(train_input=input, batch_size=1)
    assert isinstance(dm._train_ds.transform, TestTransform)

    class RandomDataset(Dataset):
        def __init__(self, size: int, length: int):
            self.len = length
            self.data = torch.ones(length, size)

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return self.len

    class TrainInputTransform(InputTransform):
        def _add_one(self, x):
            if isinstance(x, Dict):
                x["input"] += 1
            else:
                x += 1
            return x

        def per_sample_transform(self) -> Callable:
            return self._add_one

    def _add_hundred(x):
        if isinstance(x, Dict):
            x["input"] += 100
        else:
            x += 100
        return x

    dm = DataModule(
        train_input=DatasetInput(RunningStage.TRAINING, RandomDataset(64, 32), transform=TrainInputTransform),
        val_input=DatasetInput(RunningStage.TRAINING, RandomDataset(64, 32), transform=_add_hundred),
        test_input=DatasetInput(RunningStage.TRAINING, RandomDataset(64, 32)),
        batch_size=3,
    )
    batch = next(iter(dm.train_dataloader()))
    assert batch["input"][0][0] == 2
    batch = next(iter(dm.val_dataloader()))
    assert batch["input"][0][0] == 101
    batch = next(iter(dm.test_dataloader()))
    assert batch["input"][0][0] == 1
