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
from flash.core.data.data_source import DataSource
from flash.core.utilities.running_stage import RunningStage


class _AutoDatasetTestDataSource(DataSource):
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
    ds = DataSource()
    dset = BaseAutoDataset(data=dt, data_source=ds, running_stage=running_stage)
    assert dset is not None
    assert dset.running_stage == running_stage

    # check on members
    assert dset.data == dt
    assert dset.data_source == ds

    # test set the running stage
    dset.running_stage = RunningStage.PREDICTING
    assert dset.running_stage == RunningStage.PREDICTING

    # check on methods
    assert dset.load_sample is not None
    assert dset.load_sample == ds.load_sample


def test_autodataset_smoke():
    num_samples = 20
    dt = range(num_samples)
    ds = DataSource()

    dset = AutoDataset(data=dt, data_source=ds, running_stage=RunningStage.TRAINING)
    assert dset is not None
    assert dset.running_stage == RunningStage.TRAINING

    # check on members
    assert dset.data == dt
    assert dset.data_source == ds

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
    ds = DataSource()

    dset = IterableAutoDataset(data=dt, data_source=ds, running_stage=RunningStage.TRAINING)
    assert dset is not None
    assert dset.running_stage == RunningStage.TRAINING

    # check on members
    assert dset.data == dt
    assert dset.data_source == ds

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
def test_preprocessing_data_source_with_running_stage(with_dataset):
    data_source = _AutoDatasetTestDataSource(with_dataset)
    running_stage = RunningStage.TRAINING

    dataset = data_source.generate_dataset(range(10), running_stage=running_stage)

    assert len(dataset) == 10

    for idx in range(len(dataset)):
        dataset[idx]

    if with_dataset:
        assert dataset.train_load_sample_was_called
        assert dataset.train_load_data_was_called
        assert data_source.train_load_sample_with_dataset_count == len(dataset)
        assert data_source.train_load_data_with_dataset_count == 1
    else:
        assert data_source.train_load_sample_count == len(dataset)
        assert data_source.train_load_data_count == 1
