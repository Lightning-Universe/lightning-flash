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
from typing import List

import pytest

from flash.core.data.auto_dataset import AutoDataset, BaseAutoDataset, IterableAutoDataset
from flash.core.data.callback import FlashCallback
from flash.core.data.io.input import Input
from flash.core.utilities.stages import RunningStage


class _AutoDatasetTestInput(Input):
    def __init__(self, with_dset: bool):
        self._callbacks: List[FlashCallback] = []
        self.load_data_count = 0
        self.load_sample_count = 0
        self.load_sample_with_dataset_count = 0
        self.load_data_with_dataset_count = 0
        self.train_load_data_with_dataset_count = 0
        self.train_load_data_count = 0
        self.train_load_sample_with_dataset_count = 0
        self.train_load_sample_count = 0

        self.with_dset = with_dset

        if with_dset:
            self.load_data = self.load_data_with_dataset
            self.load_sample = self.load_sample_with_dataset
            self.train_load_data = self.train_load_data_with_dataset
            self.train_load_sample = self.train_load_sample_with_dataset
        else:
            self.load_data = self.load_data_no_dset
            self.load_sample = self.load_sample_no_dset
            self.train_load_data = self.train_load_data_no_dset
            self.train_load_sample = self.train_load_sample_no_dset

    def load_data_no_dset(self, data):
        self.load_data_count += 1
        return data

    def load_sample_no_dset(self, data):
        self.load_sample_count += 1
        return data

    def load_sample_with_dataset(self, data, dataset):
        self.load_sample_with_dataset_count += 1
        dataset.load_sample_was_called = True
        return data

    def load_data_with_dataset(self, data, dataset):
        self.load_data_with_dataset_count += 1
        dataset.load_data_was_called = True
        return data

    def train_load_data_no_dset(self, data):
        self.train_load_data_count += 1
        return data

    def train_load_sample_no_dset(self, data):
        self.train_load_sample_count += 1
        return data

    def train_load_sample_with_dataset(self, data, dataset):
        self.train_load_sample_with_dataset_count += 1
        dataset.train_load_sample_was_called = True
        return data

    def train_load_data_with_dataset(self, data, dataset):
        self.train_load_data_with_dataset_count += 1
        dataset.train_load_data_was_called = True
        return data


# TODO: we should test the different data types
@pytest.mark.parametrize("running_stage", [RunningStage.TRAINING, RunningStage.TESTING, RunningStage.VALIDATING])
def test_base_autodataset_smoke(running_stage):
    dt = range(10)
    ds = Input()
    dset = BaseAutoDataset(data=dt, input=ds, running_stage=running_stage)
    assert dset is not None
    assert dset.running_stage == running_stage

    # check on members
    assert dset.data == dt
    assert dset.input == ds

    # test set the running stage
    dset.running_stage = RunningStage.PREDICTING
    assert dset.running_stage == RunningStage.PREDICTING

    # check on methods
    assert dset.load_sample is not None
    assert dset.load_sample == ds.load_sample


def test_autodataset_smoke():
    num_samples = 20
    dt = range(num_samples)
    ds = Input()

    dset = AutoDataset(data=dt, input=ds, running_stage=RunningStage.TRAINING)
    assert dset is not None
    assert dset.running_stage == RunningStage.TRAINING

    # check on members
    assert dset.data == dt
    assert dset.input == ds

    # test set the running stage
    dset.running_stage = RunningStage.PREDICTING
    assert dset.running_stage == RunningStage.PREDICTING

    # check on methods
    assert dset.load_sample is not None
    assert dset.load_sample == ds.load_sample

    # check getters
    assert len(dset) == num_samples
    assert dset[0] == 0
    assert dset[9] == 9
    assert dset[11] == 11


def test_iterable_autodataset_smoke():
    num_samples = 20
    dt = range(num_samples)
    ds = Input()

    dset = IterableAutoDataset(data=dt, input=ds, running_stage=RunningStage.TRAINING)
    assert dset is not None
    assert dset.running_stage == RunningStage.TRAINING

    # check on members
    assert dset.data == dt
    assert dset.input == ds

    # test set the running stage
    dset.running_stage = RunningStage.PREDICTING
    assert dset.running_stage == RunningStage.PREDICTING

    # check on methods
    assert dset.load_sample is not None
    assert dset.load_sample == ds.load_sample

    # check getters
    itr = iter(dset)
    assert next(itr) == 0
    assert next(itr) == 1
    assert next(itr) == 2


@pytest.mark.parametrize(
    "with_dataset",
    [
        True,
        False,
    ],
)
def test_input_transforming_input_with_running_stage(with_dataset):
    input = _AutoDatasetTestInput(with_dataset)
    running_stage = RunningStage.TRAINING

    dataset = input.generate_dataset(range(10), running_stage=running_stage)

    assert len(dataset) == 10

    for idx in range(len(dataset)):
        dataset[idx]

    if with_dataset:
        assert dataset.train_load_sample_was_called
        assert dataset.train_load_data_was_called
        assert input.train_load_sample_with_dataset_count == len(dataset)
        assert input.train_load_data_with_dataset_count == 1
    else:
        assert input.train_load_sample_count == len(dataset)
        assert input.train_load_data_count == 1
