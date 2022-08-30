import pytest
import os

from flash import Trainer
from pytorch_lightning.accelerators.tpu import TPUAccelerator

@pytest.mark.skipif(not os.getenv("FLASH_RUN_TPU_TESTS", "0") == "1", reason="Should run with tpu test")
def test_tpu_trainer_single():
    trainer = Trainer(accelerator="tpu", devices=1)
    assert isinstance(trainer.accelerator, TPUAccelerator), "Expected device to be TPU"

@pytest.mark.skipif(not os.getenv("FLASH_RUN_TPU_TESTS", "0") == "1", reason="Should run with tpu test")
def test_tpu_trainer_multi_core():
    trainer = Trainer(accelerator="tpu", devices=8)
    assert isinstance(trainer.accelerator, TPUAccelerator), "Expected device to be TPU"
