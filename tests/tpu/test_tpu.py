import os
import pytest

from pytorch_lightning.accelerators.tpu import TPUAccelerator
import flash

from tests.helpers.boring_model import BoringDataModule, BoringModel


# Current state of TPU with Flash
# Single Core:
# TPU Training, Validation, and Prediction are supported.
# Multi Core:
# TPU Training, Validation are supported, but prediction is not.
@pytest.mark.skipif(not os.getenv("FLASH_RUN_TPU_TESTS", "0") == "1", reason="Should run with TPU test")
@pytest.mark.parametrize(
    "devices", (1, 8)
)
def test_tpu_finetuning(devices: int):
    boring_model = BoringModel()
    boring_dm = BoringDataModule()

    trainer = flash.Trainer(max_epochs=1, devices=devices, accelerator="tpu")
    assert isinstance(trainer.accelerator, TPUAccelerator), "Expected device to be TPU"

    trainer.finetune(model=boring_model, datamodule=boring_dm)
    assert trainer.state.finished

@pytest.mark.skipif(not os.getenv("FLASH_RUN_TPU_TESTS", "0") == "1", reason="Should run with TPU test")
@pytest.mark.parametrize(
    "devices", (1, 8)
)
def test_tpu_prediction(devices: int):
    boring_model = BoringModel()
    boring_dm = BoringDataModule()

    trainer = flash.Trainer(fast_dev_run=True, devices=devices, accelerator="tpu")
    assert isinstance(trainer.accelerator, TPUAccelerator), "Expected device to be TPU"

    trainer.fit(model=boring_model, datamodule=boring_dm)
    trainer.validate(model=boring_model, datamodule=boring_dm)
    trainer.test(model=boring_model, datamodule=boring_dm)
    predictions = None
    if devices > 1:
        with pytest.raises(NotImplementedError, match="not supported"):
            predictions = trainer.predict(model=boring_model, datamodule=boring_dm)
    else:
        predictions = trainer.predict(model=boring_model, datamodule=boring_dm)

    assert predictions is not None and len(predictions) != 0, "Prediction not successful"
