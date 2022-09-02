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
import os

import pytest
import torch.nn.functional as F
from pytorch_lightning.accelerators.tpu import TPUAccelerator
from torch.utils.data import DataLoader

import flash
from tests.core.test_finetuning import DummyDataset, TestTaskWithFinetuning
from tests.helpers.boring_model import BoringDataModule, BoringModel

# Current state of TPU with Flash (as of v0.8 release)
# Single Core:
# TPU Training, Validation, and Prediction are supported.
# Multi Core:
# TPU Training, Validation are supported, but prediction is not.

# Helper function
def _assert_state_finished(trainer, fn_name):
    assert trainer.state.finished and trainer.state.fn == fn_name


@pytest.mark.skipif(not os.getenv("FLASH_RUN_TPU_TESTS", "0") == "1", reason="Should run with TPU test")
@pytest.mark.parametrize("devices", (1, 8))
def test_tpu_finetuning(devices: int):
    task = TestTaskWithFinetuning(loss_fn=F.nll_loss)

    trainer = flash.Trainer(max_epochs=1, devices=devices, accelerator="tpu")
    assert isinstance(trainer.accelerator, TPUAccelerator)

    ds = DummyDataset()
    trainer.finetune(model=task, train_dataloader=DataLoader(ds))
    _assert_state_finished(trainer, "fit")


@pytest.mark.skipif(not os.getenv("FLASH_RUN_TPU_TESTS", "0") == "1", reason="Should run with TPU test")
@pytest.mark.parametrize("devices", (1, 8))
def test_tpu_prediction(devices: int):
    boring_model = BoringModel()
    boring_dm = BoringDataModule()

    trainer = flash.Trainer(fast_dev_run=True, devices=devices, accelerator="tpu")
    assert isinstance(trainer.accelerator, TPUAccelerator)

    trainer.fit(model=boring_model, datamodule=boring_dm)
    _assert_state_finished(trainer, "fit")
    trainer.validate(model=boring_model, datamodule=boring_dm)
    _assert_state_finished(trainer, "validate")
    trainer.test(model=boring_model, datamodule=boring_dm)
    _assert_state_finished(trainer, "test")

    if devices > 1:
        with pytest.raises(NotImplementedError, match="not supported"):
            predictions = trainer.predict(model=boring_model, datamodule=boring_dm)
        return

    predictions = trainer.predict(model=boring_model, datamodule=boring_dm)
    assert predictions is not None and len(predictions) != 0, "Prediction not successful"
    _assert_state_finished(trainer, "predict")
