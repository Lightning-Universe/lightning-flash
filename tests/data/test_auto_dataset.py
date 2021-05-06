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
from typing import Any, Dict, List

import pytest
from pytorch_lightning.trainer.states import RunningStage

from flash.data.auto_dataset import AutoDataset
from flash.data.callback import FlashCallback
from flash.data.data_pipeline import DataPipeline
from flash.data.data_source import DataSource
from flash.data.process import Preprocess


class _AutoDatasetTestPreprocess(Preprocess):

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

    def get_state_dict(self) -> Dict[str, Any]:
        return {"with_dset": self.with_dset}

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool):
        return _AutoDatasetTestPreprocess(state_dict["with_dset"])

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
def test_autodataset_smoke(running_stage):
    dt = range(10)
    ds = DataSource()
    dset = AutoDataset(data=dt, data_source=ds, running_stage=running_stage)
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
    pass


@pytest.mark.parametrize(
    "with_dataset,with_running_stage",
    [
        (True, False),
        (True, True),
        (False, False),
        (False, True),
    ],
)
def test_autodataset_with_functions(
    with_dataset: bool,
    with_running_stage: bool,
):

    functions = _AutoDatasetTestPreprocess(with_dataset)

    load_sample_func = functions.load_sample
    load_data_func = functions.load_data

    if with_running_stage:
        running_stage = RunningStage.TRAINING
    else:
        running_stage = None
    dset = AutoDataset(
        range(10),
        load_data=load_data_func,
        load_sample=load_sample_func,
        running_stage=running_stage,
    )

    assert len(dset) == 10

    for idx in range(len(dset)):
        dset[idx]

    if with_dataset:
        assert dset.load_sample_was_called
        assert dset.load_data_was_called
        assert functions.load_sample_with_dataset_count == len(dset)
        assert functions.load_data_with_dataset_count == 1
    else:
        assert functions.load_data_count == 1
        assert functions.load_sample_count == len(dset)


def test_autodataset_warning():
    with pytest.warns(
        UserWarning, match="``datapipeline`` is specified but load_sample and/or load_data are also specified"
    ):
        AutoDataset(range(10), load_data=lambda x: x, load_sample=lambda x: x, data_pipeline=DataPipeline())


@pytest.mark.parametrize(
    "with_dataset",
    [
        True,
        False,
    ],
)
def test_preprocessing_data_pipeline_with_running_stage(with_dataset):
    pipe = DataPipeline(_AutoDatasetTestPreprocess(with_dataset))

    running_stage = RunningStage.TRAINING

    dataset = pipe._generate_auto_dataset(range(10), running_stage=running_stage)

    assert len(dataset) == 10

    for idx in range(len(dataset)):
        dataset[idx]

    if with_dataset:
        assert dataset.train_load_sample_was_called
        assert dataset.train_load_data_was_called
        assert pipe._preprocess_pipeline.train_load_sample_with_dataset_count == len(dataset)
        assert pipe._preprocess_pipeline.train_load_data_with_dataset_count == 1
    else:
        assert pipe._preprocess_pipeline.train_load_sample_count == len(dataset)
        assert pipe._preprocess_pipeline.train_load_data_count == 1


@pytest.mark.parametrize(
    "with_dataset",
    [
        True,
        False,
    ],
)
def test_preprocessing_data_pipeline_no_running_stage(with_dataset):
    pipe = DataPipeline(_AutoDatasetTestPreprocess(with_dataset))

    dataset = pipe._generate_auto_dataset(range(10), running_stage=None)

    with pytest.raises(RuntimeError, match='`__len__` for `load_sample`'):
        for idx in range(len(dataset)):
            dataset[idx]

    # will be triggered when running stage is set
    if with_dataset:
        assert not hasattr(dataset, 'load_sample_was_called')
        assert not hasattr(dataset, 'load_data_was_called')
        assert pipe._preprocess_pipeline.load_sample_with_dataset_count == 0
        assert pipe._preprocess_pipeline.load_data_with_dataset_count == 0
    else:
        assert pipe._preprocess_pipeline.load_sample_count == 0
        assert pipe._preprocess_pipeline.load_data_count == 0

    dataset.running_stage = RunningStage.TRAINING

    if with_dataset:
        assert pipe._preprocess_pipeline.train_load_data_with_dataset_count == 1
        assert dataset.train_load_data_was_called
    else:
        assert pipe._preprocess_pipeline.train_load_data_count == 1
