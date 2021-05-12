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
import numpy as np
import pytest
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from flash.data.data_module import DataModule
from flash.data.splits import SplitDataset


def test_split_dataset(tmpdir):

    train_ds, val_ds = DataModule._split_train_val(range(100), val_split=0.1)
    assert len(train_ds) == 90
    assert len(val_ds) == 10
    assert len(np.unique(train_ds.indices)) == len(train_ds.indices)

    with pytest.raises(MisconfigurationException, match="[0, 99]"):
        SplitDataset(range(100), indices=[100])

    with pytest.raises(MisconfigurationException, match="[0, 49]"):
        SplitDataset(range(50), indices=[-1])

    with pytest.raises(MisconfigurationException, match="[0, 49]"):
        SplitDataset(list(range(50)) + list(range(50)), indices=[-1])

    with pytest.raises(MisconfigurationException, match="[0, 99]"):
        SplitDataset(list(range(50)) + list(range(50)), indices=[-1], use_duplicated_indices=True)

    class Dataset:

        def __init__(self):
            self.data = [0, 1, 2]
            self.name = "something"

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return len(self.data)

    split_dataset = SplitDataset(Dataset(), indices=[0])
    assert split_dataset.name == "something"

    assert split_dataset._INTERNAL_KEYS == ("dataset", "indices", "data")

    split_dataset.is_passed_down = True
    assert split_dataset.dataset.is_passed_down
