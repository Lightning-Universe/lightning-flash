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
import torch
from pytorch_lightning import Callback
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from flash import Trainer
from flash.core.utilities.imports import _TORCH_ORT_AVAILABLE
from flash.text import TextClassifier
from flash.text.ort_callback import ORTCallback
from tests.helpers.boring_model import BoringModel
from tests.helpers.utils import _TEXT_TESTING
from tests.text.classification.test_model import DummyDataset, TEST_HF_BACKBONE

if _TORCH_ORT_AVAILABLE:
    from torch_ort import ORTModule


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.skipif(not _TORCH_ORT_AVAILABLE, reason="ORT Module aren't installed.")
def test_init_train_enable_ort(tmpdir):
    class TestCallback(Callback):
        def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
            assert isinstance(pl_module.model, ORTModule)

    model = TextClassifier(2, TEST_HF_BACKBONE, enable_ort=True)
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, callbacks=TestCallback())
    trainer.fit(
        model,
        train_dataloader=torch.utils.data.DataLoader(DummyDataset()),
        val_dataloaders=torch.utils.data.DataLoader(DummyDataset()),
    )
    trainer.test(model, test_dataloaders=torch.utils.data.DataLoader(DummyDataset()))


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TORCH_ORT_AVAILABLE, reason="ORT Module aren't installed.")
def test_ort_callback_fails_no_model(tmpdir):
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, callbacks=ORTCallback())
    with pytest.raises(MisconfigurationException, match="Torch ORT requires to wrap a single model"):
        trainer.fit(
            model,
            train_dataloader=torch.utils.data.DataLoader(DummyDataset()),
            val_dataloaders=torch.utils.data.DataLoader(DummyDataset()),
        )
