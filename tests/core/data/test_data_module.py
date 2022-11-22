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
from unittest import mock

import numpy as np
import pytest
import torch
from pytorch_lightning import seed_everything
from torch.utils.data import Dataset

from flash import Task, Trainer
from flash.core.data.data_module import DataModule, DatasetInput
from flash.core.data.io.input import Input
from flash.core.data.io.input_transform import InputTransform
from flash.core.utilities.imports import _CORE_TESTING, _IMAGE_TESTING, _TORCHVISION_AVAILABLE
from flash.core.utilities.stages import RunningStage
from tests.helpers.boring_model import BoringModel

if _TORCHVISION_AVAILABLE:
    import torchvision.transforms as T


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
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

        def train_per_batch_transform_on_device(self) -> Callable:
            return train_fn

        def val_per_batch_transform_on_device(self) -> Callable:
            return val_fn

        def test_per_batch_transform_on_device(self) -> Callable:
            return test_fn

        def predict_per_batch_transform_on_device(self) -> Callable:
            return predict_fn

    transform = TestTransform()
    assert transform._transform is not None

    train_dataset = Input(RunningStage.TRAINING, np.arange(10, dtype=np.float32))
    assert train_dataset.running_stage == RunningStage.TRAINING

    val_dataset = Input(RunningStage.VALIDATING, np.arange(10, dtype=np.float32))
    assert val_dataset.running_stage == RunningStage.VALIDATING

    test_dataset = Input(RunningStage.TESTING, np.arange(10, dtype=np.float32))
    assert test_dataset.running_stage == RunningStage.TESTING

    predict_dataset = Input(RunningStage.PREDICTING, np.arange(10, dtype=np.float32))
    assert predict_dataset.running_stage == RunningStage.PREDICTING

    dm = DataModule(
        train_input=train_dataset,
        val_input=val_dataset,
        test_input=test_dataset,
        predict_input=predict_dataset,
        transform=transform,
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

        def predict_step(self, batch, *args, **kwargs):
            assert sum(batch > 500) == 2
            assert torch.equal(batch, torch.tensor([1000.0, 1001.0]))

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
    trainer.fit(model, datamodule=dm)
    trainer.validate(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
    trainer.predict(model, datamodule=dm)

    # Test that plain lightning module works with FlashDataModule
    class SampleBoringModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 1)

    model = SampleBoringModel()
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model, datamodule=dm)
    trainer.validate(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
    trainer.predict(model, datamodule=dm)

    transform = TestTransform()
    input = Input(RunningStage.TRAINING)
    dm = DataModule(train_input=input, batch_size=1, transform=transform)
    assert isinstance(dm.input_transform, TestTransform)

    class RandomDataset(Dataset):
        def __init__(self, size: int, length: int):
            self.len = length
            self.data = torch.ones(length, size)

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return self.len

    def _add_hundred(x):
        if isinstance(x, Dict):
            x["input"] += 100
        else:
            x += 100
        return x

    class TrainInputTransform(InputTransform):
        def _add_one(self, x):
            if isinstance(x, Dict):
                x["input"] += 1
            else:
                x += 1
            return x

        def per_sample_transform(self) -> Callable:
            return self._add_one

        def val_per_sample_transform(self) -> Callable:
            return _add_hundred

    dm = DataModule(
        train_input=DatasetInput(RunningStage.TRAINING, RandomDataset(64, 32)),
        val_input=DatasetInput(RunningStage.VALIDATING, RandomDataset(64, 32)),
        test_input=DatasetInput(RunningStage.TESTING, RandomDataset(64, 32)),
        batch_size=3,
        transform=TrainInputTransform(),
    )
    batch = next(iter(dm.train_dataloader()))
    assert batch["input"][0][0] == 2
    batch = next(iter(dm.val_dataloader()))
    assert batch["input"][0][0] == 101
    batch = next(iter(dm.test_dataloader()))
    assert batch["input"][0][0] == 2


class TestInput(Input):
    def train_load_data(self, _):
        return [(0, 1, 2, 3), (0, 1, 2, 3)]

    def val_load_data(self, _):
        self.val_load_sample_called = False
        return list(range(5))

    def val_load_sample(self, sample):
        self.val_load_sample_called = True
        return {"a": sample, "b": sample + 1}

    def test_load_data(self, _):
        return [[torch.rand(1), torch.rand(1)], [torch.rand(1), torch.rand(1)]]


@dataclass
class TestInputTransform(InputTransform):
    train_per_sample_transform_called = False
    train_collate_called = False
    train_per_batch_transform_on_device_called = False
    val_per_sample_transform_called = False
    val_collate_called = False
    val_per_batch_transform_on_device_called = False
    test_per_sample_transform_called = False

    def _train_per_sample_transform(self, sample):
        self.train_per_sample_transform_called = True
        return sample + (5,)

    def train_per_sample_transform(self):
        return self._train_per_sample_transform

    def _train_collate(self, samples):
        self.train_collate_called = True
        return torch.tensor([list(s) for s in samples])

    def train_collate(self):
        return self._train_collate

    def _train_per_batch_transform_on_device(self, batch):
        self.train_per_batch_transform_on_device_called = True
        assert torch.equal(batch, torch.tensor([[0, 1, 2, 3, 5], [0, 1, 2, 3, 5]]))

    def train_per_batch_transform_on_device(self):
        return self._train_per_batch_transform_on_device

    def _val_per_sample_transform(self, sample):
        self.val_per_sample_transform_called = True
        return sample

    def val_per_sample_transform(self):
        return self._val_per_sample_transform

    def _val_collate(self, samples):
        self.val_collate_called = True
        _count = samples[0]["a"]
        assert samples == [{"a": _count, "b": _count + 1}, {"a": _count + 1, "b": _count + 2}]
        return {"a": torch.tensor([0, 1]), "b": torch.tensor([1, 2])}

    def val_collate(self):
        return self._val_collate

    def _val_per_batch_transform_on_device(self, batch):
        self.val_per_batch_transform_on_device_called = True
        if isinstance(batch, list):
            batch = batch[0]
        assert torch.equal(batch["a"], torch.tensor([0, 1]))
        assert torch.equal(batch["b"], torch.tensor([1, 2]))
        return [False]

    def val_per_batch_transform_on_device(self):
        return self._val_per_batch_transform_on_device

    def _test_per_sample_transform(self, sample):
        self.test_per_sample_transform_called = True
        return sample

    def test_per_sample_transform(self):
        return self._test_per_sample_transform


class TestInputTransform2(TestInputTransform):
    def _val_per_sample_transform(self, sample):
        self.val_per_sample_transform_called = True
        return {"a": torch.tensor(sample["a"]), "b": torch.tensor(sample["b"])}


class CustomModel(Task):
    def __init__(self):
        super().__init__(model=torch.nn.Linear(1, 1), loss_fn=torch.nn.MSELoss())

    def training_step(self, batch, batch_idx):
        assert batch is None

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, list):
            batch = batch[0]
        assert batch is False

    def test_step(self, batch, batch_idx):
        assert len(batch) == 2
        assert batch[0].shape == torch.Size([2, 1])


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
def test_transformations(tmpdir):
    transform = TestInputTransform()
    datamodule = DataModule(
        TestInput(RunningStage.TRAINING, [1]),
        TestInput(RunningStage.VALIDATING, [1]),
        TestInput(RunningStage.TESTING, [1]),
        transform=transform,
        batch_size=2,
        num_workers=0,
    )

    assert datamodule.train_dataloader().dataset[0] == (0, 1, 2, 3)
    batch = next(iter(datamodule.train_dataloader()))
    assert torch.equal(batch, torch.tensor([[0, 1, 2, 3, 5], [0, 1, 2, 3, 5]]))

    assert datamodule.val_dataloader().dataset[0] == {"a": 0, "b": 1}
    assert datamodule.val_dataloader().dataset[1] == {"a": 1, "b": 2}
    batch = next(iter(datamodule.val_dataloader()))

    datamodule = DataModule(
        TestInput(RunningStage.TRAINING, [1]),
        TestInput(RunningStage.VALIDATING, [1]),
        TestInput(RunningStage.TESTING, [1]),
        transform=TestInputTransform2,
        batch_size=2,
        num_workers=0,
    )
    batch = next(iter(datamodule.val_dataloader()))
    assert torch.equal(batch["a"], torch.tensor([0, 1]))
    assert torch.equal(batch["b"], torch.tensor([1, 2]))

    model = CustomModel()
    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        limit_test_batches=2,
        limit_predict_batches=2,
        num_sanity_val_steps=1,
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

    assert datamodule.input_transform.train_per_sample_transform_called
    assert datamodule.input_transform.train_collate_called
    assert datamodule.input_transform.train_per_batch_transform_on_device_called
    assert datamodule.input_transform.train_per_sample_transform_called
    assert datamodule.input_transform.val_collate_called
    assert datamodule.input_transform.val_per_batch_transform_on_device_called
    assert datamodule.input_transform.test_per_sample_transform_called


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_datapipeline_transformations_overridden_by_task():
    # define input transforms
    class ImageInput(Input):
        def load_data(self, folder):
            # from folder -> return files paths
            return ["a.jpg", "b.jpg"]

        def load_sample(self, path):
            # from a file path, load the associated image
            return np.random.uniform(0, 1, (64, 64, 3))

    class ImageClassificationInputTransform(InputTransform):
        def per_sample_transform(self) -> Callable:
            return T.Compose([T.ToTensor()])

        def per_batch_transform_on_device(self) -> Callable:
            return T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    class OverrideInputTransform(InputTransform):
        def per_sample_transform(self) -> Callable:
            return T.Compose([T.ToTensor(), T.Resize(128)])

    # define task which overrides transforms using set_state
    class CustomModel(Task):
        def __init__(self):
            super().__init__(model=torch.nn.Linear(1, 1), loss_fn=torch.nn.MSELoss())

            # override default transform to resize images
            self.input_transform = OverrideInputTransform

        def training_step(self, batch, batch_idx):
            assert batch.shape == torch.Size([2, 3, 128, 128])
            assert torch.max(batch) <= 1.0
            assert torch.min(batch) >= 0.0

        def validation_step(self, batch, batch_idx):
            assert batch.shape == torch.Size([2, 3, 128, 128])
            assert torch.max(batch) <= 1.0
            assert torch.min(batch) >= 0.0

    transform = ImageClassificationInputTransform()
    datamodule = DataModule(
        ImageInput(RunningStage.TRAINING, [1]),
        ImageInput(RunningStage.VALIDATING, [1]),
        transform=transform,
        batch_size=2,
        num_workers=0,
    )

    # call trainer
    model = CustomModel()
    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        num_sanity_val_steps=1,
    )
    trainer.fit(model, datamodule=datamodule)


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
@pytest.mark.parametrize("sampler, callable", [(mock.MagicMock(), True), (mock.NonCallableMock(), False)])
@mock.patch("flash.core.data.data_module.DataLoader")
def test_dataloaders_with_sampler(mock_dataloader, sampler, callable):
    train_input = TestInput(RunningStage.TRAINING, [1])
    datamodule = DataModule(
        train_input,
        TestInput(RunningStage.VALIDATING, [1]),
        TestInput(RunningStage.TESTING, [1]),
        batch_size=2,
        num_workers=0,
        sampler=sampler,
    )

    assert datamodule.sampler is sampler
    dl = datamodule.train_dataloader()

    if callable:
        sampler.assert_called_once_with(train_input)

    kwargs = mock_dataloader.call_args[1]
    assert "sampler" in kwargs
    assert kwargs["sampler"] is (sampler.return_value if callable else sampler)
    for dl in [datamodule.val_dataloader(), datamodule.test_dataloader()]:
        kwargs = mock_dataloader.call_args[1]
        assert "sampler" not in kwargs


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
def test_val_split():
    datamodule = DataModule(
        Input(RunningStage.TRAINING, [1] * 100),
        batch_size=2,
        num_workers=0,
        val_split=0.2,
    )

    assert len(datamodule.train_dataset) == 80
    assert len(datamodule.val_dataset) == 20
