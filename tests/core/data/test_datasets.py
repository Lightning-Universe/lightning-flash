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
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from flash.core.data.datasets import FlashDataset, FlashIterableDataset
from flash.core.utilities.stages import RunningStage


def test_flash_dataset():
    class TestDataset(FlashDataset):

        pass

    with pytest.raises(MisconfigurationException, match="You should provide a running_stage"):
        _ = TestDataset.from_data(None)

    dataset = TestDataset.from_data(None, running_stage=RunningStage.TRAINING)
    assert isinstance(dataset, TestDataset)
    assert dataset.data is None

    dataset = TestDataset.from_data(range(10), running_stage=RunningStage.TRAINING)
    assert len(dataset) == 10
    assert dataset[0] == 0

    class TestDataset(FlashDataset):
        def load_data(self, data, data2):
            return [(x, y) for x, y in zip(data, data2)]

    dataset = TestDataset.from_data(range(10), range(10, 20), running_stage=RunningStage.TRAINING)
    assert len(dataset) == 10
    assert dataset[0] == (0, 10)
    assert dataset[-1] == (9, 19)

    class TestDataset(FlashDataset):
        def __init__(self, running_stage: RunningStage, shift=0):
            super().__init__(running_stage)
            self.shift = shift

        def load_data(self, data, data2):
            shift = self.shift if self.training else -self.shift
            return [(x, y + shift) for x, y in zip(data, data2)]

        def test_load_sample(self, sample):
            return list(sample)

        def val_load_data(self, data, *args):
            return data

        def predict_load_data(self, data, data2):
            return [{"input": x, "target": y} for x, y in zip(data, data2)]

    dataset = TestDataset(running_stage=RunningStage.TRAINING, shift=1)
    dataset.pass_args_to_load_data(range(10), range(10, 20))
    assert len(dataset) == 10
    assert dataset[0] == (0, 11)
    assert dataset[-1] == (9, 20)

    dataset = TestDataset(running_stage=RunningStage.VALIDATING, shift=1)
    dataset.pass_args_to_load_data(range(10), range(10, 20))
    assert len(dataset) == 10
    assert dataset[0] == 0
    assert dataset[-1] == 9

    dataset = TestDataset(running_stage=RunningStage.TESTING, shift=1)
    dataset.pass_args_to_load_data(range(10), range(10, 20))
    assert len(dataset) == 10
    assert dataset[0] == [0, 9]
    assert dataset[-1] == [9, 18]

    dataset = TestDataset(running_stage=RunningStage.PREDICTING, shift=1)
    dataset.pass_args_to_load_data(range(10), range(10, 20))
    assert dataset[0] == {"input": 0, "target": 10}

    class TestDataset(FlashIterableDataset):
        def __init__(self, running_stage: RunningStage, shift=0):
            super().__init__(running_stage)
            self.shift = shift

        def load_data(self, data, data2):
            shift = self.shift if self.training else -self.shift
            return enumerate([(x, y + shift) for x, y in zip(data, data2)])

        def val_load_data(self, data, *args):
            return enumerate(data)

        def load_sample(self, data):
            return (data[0], (data[1][0] + 2,))

    dataset = TestDataset(running_stage=RunningStage.TRAINING, shift=1)
    dataset.pass_args_to_load_data(range(10), range(10, 20))
    dataset_iter = iter(dataset)
    assert next(dataset_iter) == (0, (2,))
