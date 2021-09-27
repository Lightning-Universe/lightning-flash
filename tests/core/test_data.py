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

from flash import DataModule

# ======== Mock functions ========


class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        return torch.rand(1, 28, 28), torch.randint(10, size=(1,)).item()

    def __len__(self) -> int:
        return 10


# ===============================


def test_init():
    train_ds, val_ds, test_ds = DummyDataset(), DummyDataset(), DummyDataset()
    DataModule(train_ds)
    DataModule(train_ds, val_ds)
    DataModule(train_ds, val_ds, test_ds)
    assert DataModule().data_pipeline


def test_dataloaders():
    train_ds, val_ds, test_ds = DummyDataset(), DummyDataset(), DummyDataset()
    dm = DataModule(train_ds, val_ds, test_ds, num_workers=0)
    for dl in [
        dm.train_dataloader(),
        dm.val_dataloader(),
        dm.test_dataloader(),
    ]:
        x, y = next(iter(dl))
        assert x.shape == (4, 1, 28, 28)


def test_cpu_count_none():
    train_ds = DummyDataset()
    dm = DataModule(train_ds, num_workers=None)
    assert dm.num_workers == 0
