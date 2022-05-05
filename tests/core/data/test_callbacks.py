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

import pytest
import torch
from torch import tensor

from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_module import DataModule
from flash.core.data.io.input import Input
from flash.core.data.io.input_transform import InputTransform
from flash.core.utilities.imports import _CORE_TESTING
from flash.core.utilities.stages import RunningStage


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
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
                Input(RunningStage.TRAINING, train_data),
                Input(RunningStage.VALIDATING, val_data),
                Input(RunningStage.TESTING, test_data),
                Input(RunningStage.PREDICTING, predict_data),
                transform=InputTransform(),
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
