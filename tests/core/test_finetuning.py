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
import pytest
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.nn import functional as F

from flash import ClassificationTask, Trainer
from flash.core.finetuning import FlashBaseFinetuning
from tests.core.test_model import DummyDataset


@pytest.mark.parametrize(
    "strategy", ['no_freeze', 'freeze', 'freeze_unfreeze', 'unfreeze_milestones', None, 'cls', 'chocolat']
)
def test_finetuning(tmpdir: str, strategy):
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10), nn.LogSoftmax())
    train_dl = torch.utils.data.DataLoader(DummyDataset())
    val_dl = torch.utils.data.DataLoader(DummyDataset())
    task = ClassificationTask(model, F.nll_loss)
    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    if strategy == "cls":
        strategy = FlashBaseFinetuning()
    if strategy == 'chocolat' or strategy is None:
        with pytest.raises(MisconfigurationException, match="strategy should be provided"):
            trainer.finetune(task, train_dl, val_dl, strategy=strategy)
    else:
        trainer.finetune(task, train_dl, val_dl, strategy=strategy)
