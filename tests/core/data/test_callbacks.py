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
from typing import Any

import torch
from torch import tensor

from flash.core.data.callback import BaseDataFetcher
from flash.core.data.input_transform import InputTransform
from flash.core.data.io.input_base import Input
from flash.core.data.new_data_module import DataModule
from flash.core.utilities.stages import RunningStage


def test_base_data_fetcher(tmpdir):
    class CheckData(BaseDataFetcher):
        def check(self):
            assert self.batches["val"]["load_sample"] == [0, 1, 2, 3, 4]
            assert self.batches["val"]["per_sample_transform"] == [0, 1, 2, 3, 4]
            assert torch.equal(self.batches["val"]["collate"][0], tensor([0, 1, 2, 3, 4]))
            assert torch.equal(self.batches["val"]["per_batch_transform"][0], tensor([0, 1, 2, 3, 4]))
            assert self.batches["train"] == {}
            assert self.batches["test"] == {}
            assert self.batches["predict"] == {}

    class CustomDataModule(DataModule):
        @staticmethod
        def configure_data_fetcher():
            return CheckData()

        @classmethod
        def from_inputs(cls, train_data: Any, val_data: Any, test_data: Any, predict_data: Any) -> "CustomDataModule":
            return cls(
                Input(RunningStage.TRAINING, train_data, transform=InputTransform),
                Input(RunningStage.VALIDATING, val_data, transform=InputTransform),
                Input(RunningStage.TESTING, test_data, transform=InputTransform),
                Input(RunningStage.PREDICTING, predict_data, transform=InputTransform),
                batch_size=5,
            )

    dm = CustomDataModule.from_inputs(range(5), range(5), range(5), range(5))
    data_fetcher: CheckData = dm.data_fetcher

    if not hasattr(dm, "_val_iter"):
        dm._reset_iterator("val")

    with data_fetcher.enable():
        assert data_fetcher.enabled
        _ = next(dm._val_iter)

    data_fetcher.check()
    data_fetcher.reset()
    assert data_fetcher.batches == {"train": {}, "test": {}, "val": {}, "predict": {}, "serve": {}}


def test_data_loaders_num_workers_to_0(tmpdir):
    """num_workers should be set to `0` internally for visualization and not for training."""

    datamodule = DataModule(train_dataset=range(10), num_workers=3)
    iterator = datamodule._reset_iterator(RunningStage.TRAINING)
    assert isinstance(iterator, torch.utils.data.dataloader._SingleProcessDataLoaderIter)
    iterator = iter(datamodule.train_dataloader())
    assert isinstance(iterator, torch.utils.data.dataloader._MultiProcessingDataLoaderIter)
    assert datamodule.num_workers == 3
