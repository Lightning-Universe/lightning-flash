import pytest
from pytorch_lightning.trainer.states import RunningStage

from flash.data.auto_dataset import AutoDataset
from flash.data.data_pipeline import DataPipeline
from flash.data.process import Postprocess, Preprocess


class _AutoDatasetTestPreprocess(Preprocess):

    def __init__(self, with_dset: bool):
        self.load_data_count = 0
        self.load_sample_count = 0
        self.load_sample_with_dataset_count = 0
        self.load_data_with_dataset_count = 0
        self.train_load_data_with_dataset_count = 0
        self.train_load_data_count = 0
        self.train_load_sample_with_dataset_count = 0
        self.train_load_sample_count = 0

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
        _ = dset[idx]

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
        UserWarning,
        match="``datapipeline`` is specified but load_sample and/or load_data are also specified. Won't use datapipeline"
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
        _ = dataset[idx]

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

    with pytest.raises(
        RuntimeError,
        match='Names for LoadSample and LoadData could not be inferred. Consider setting the RunningStage'
    ):
        for idx in range(len(dataset)):
            _ = dataset[idx]

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
