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
import torch

from flash import DataKeys, DataModule, RunningStage
from flash.core.data.data_module import DatasetInput

# ======== Mock functions ========


class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        return torch.rand(1, 28, 28), torch.randint(10, size=(1,)).item()

    def __len__(self) -> int:
        return 10


# ===============================


def test_init():
    train_input = DatasetInput(RunningStage.TRAINING, DummyDataset())
    val_input = DatasetInput(RunningStage.VALIDATING, DummyDataset())
    test_input = DatasetInput(RunningStage.TESTING, DummyDataset())
    DataModule(train_input, batch_size=1)
    DataModule(train_input, val_input, batch_size=1)
    DataModule(train_input, val_input, test_input, batch_size=1)
    assert DataModule(batch_size=1).data_pipeline


def test_dataloaders():
    train_input = DatasetInput(RunningStage.TRAINING, DummyDataset())
    val_input = DatasetInput(RunningStage.VALIDATING, DummyDataset())
    test_input = DatasetInput(RunningStage.TESTING, DummyDataset())
    dm = DataModule(train_input, val_input, test_input, num_workers=0, batch_size=1)
    for dl in [
        dm.train_dataloader(),
        dm.val_dataloader(),
        dm.test_dataloader(),
    ]:
        x = next(iter(dl))[DataKeys.INPUT]
        assert x.shape == (1, 1, 28, 28)


def test_cpu_count_none():
    train_input = DatasetInput(RunningStage.TRAINING, DummyDataset())
    dm = DataModule(train_input, num_workers=None, batch_size=1)
    assert dm.num_workers == 0
