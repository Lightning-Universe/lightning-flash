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
from unittest.mock import ANY, MagicMock, call, patch

import pytest
import torch
from flash import DataKeys
from flash.core.data.data_module import DataModule, DatasetInput
from flash.core.data.io.input_transform import InputTransform
from flash.core.model import Task
from flash.core.trainer import Trainer
from flash.core.utilities.imports import _TOPIC_CORE_AVAILABLE
from flash.core.utilities.stages import RunningStage


@pytest.mark.skipif(not _TOPIC_CORE_AVAILABLE, reason="Not testing core.")
@patch("pickle.dumps")  # need to mock pickle or we get pickle error
@patch("torch.save")  # need to mock torch.save, or we get pickle error
def test_flash_callback(_, __, tmpdir):  # noqa: PT019
    """Test the callback hook system for fit."""

    callback_mock = MagicMock()

    inputs = [(torch.rand(1), torch.rand(1))]
    transform = InputTransform()
    dm = DataModule(
        DatasetInput(RunningStage.TRAINING, inputs),
        DatasetInput(RunningStage.VALIDATING, inputs),
        DatasetInput(RunningStage.TESTING, inputs),
        transform=transform,
        batch_size=1,
        num_workers=0,
        data_fetcher=callback_mock,
    )

    _ = next(iter(dm.train_dataloader()))

    assert callback_mock.method_calls == [
        call.on_load_sample(ANY, RunningStage.TRAINING),
        call.on_per_sample_transform(ANY, RunningStage.TRAINING),
        call.on_collate(ANY, RunningStage.TRAINING),
        call.on_per_batch_transform(ANY, RunningStage.TRAINING),
    ]

    class CustomModel(Task):
        def __init__(self):
            super().__init__(model=torch.nn.Linear(1, 1), loss_fn=torch.nn.MSELoss())

        def training_step(self, batch, batch_idx):
            batch = (batch[DataKeys.INPUT], batch[DataKeys.TARGET])
            return super().training_step(batch, batch_idx)

        def validation_step(self, batch, batch_idx):
            batch = (batch[DataKeys.INPUT], batch[DataKeys.TARGET])
            return super().validation_step(batch, batch_idx)

        def test_step(self, batch, batch_idx):
            batch = (batch[DataKeys.INPUT], batch[DataKeys.TARGET])
            return super().test_step(batch, batch_idx)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=1,
        limit_train_batches=1,
    )
    transform = InputTransform()
    dm = DataModule(
        DatasetInput(RunningStage.TRAINING, inputs),
        DatasetInput(RunningStage.VALIDATING, inputs),
        DatasetInput(RunningStage.TESTING, inputs),
        transform=transform,
        batch_size=1,
        num_workers=0,
        data_fetcher=callback_mock,
    )
    trainer.fit(CustomModel(), datamodule=dm)

    assert callback_mock.method_calls == [
        call.on_load_sample(ANY, RunningStage.TRAINING),
        call.on_per_sample_transform(ANY, RunningStage.TRAINING),
        call.on_collate(ANY, RunningStage.TRAINING),
        call.on_per_batch_transform(ANY, RunningStage.TRAINING),
        call.on_load_sample(ANY, RunningStage.VALIDATING),
        call.on_per_sample_transform(ANY, RunningStage.VALIDATING),
        call.on_collate(ANY, RunningStage.VALIDATING),
        call.on_per_batch_transform(ANY, RunningStage.VALIDATING),
        call.on_per_batch_transform_on_device(ANY, RunningStage.VALIDATING),
        call.on_load_sample(ANY, RunningStage.TRAINING),
        call.on_per_sample_transform(ANY, RunningStage.TRAINING),
        call.on_collate(ANY, RunningStage.TRAINING),
        call.on_per_batch_transform(ANY, RunningStage.TRAINING),
        call.on_per_batch_transform_on_device(ANY, RunningStage.TRAINING),
        call.on_load_sample(ANY, RunningStage.VALIDATING),
        call.on_per_sample_transform(ANY, RunningStage.VALIDATING),
        call.on_collate(ANY, RunningStage.VALIDATING),
        call.on_per_batch_transform(ANY, RunningStage.VALIDATING),
        call.on_per_batch_transform_on_device(ANY, RunningStage.VALIDATING),
    ]
