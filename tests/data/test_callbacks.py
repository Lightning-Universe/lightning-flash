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

from pathlib import Path
from typing import Any, List, Sequence

import numpy as np
import torch
from PIL import Image
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer.states import RunningStage
from torch import tensor

from flash.data.base_viz import BaseVisualization
from flash.data.callback import BaseDataFetcher
from flash.data.data_module import DataModule
from flash.data.utils import _STAGES_PREFIX
from flash.vision import ImageClassificationData


def _rand_image():
    return Image.fromarray(np.random.randint(0, 255, (196, 196, 3), dtype="uint8"))


def test_base_data_fetcher(tmpdir):

    class CheckData(BaseDataFetcher):

        def check(self):
            assert self.batches["val"]["load_sample"] == [0, 1, 2, 3, 4]
            assert self.batches["val"]["pre_tensor_transform"] == [0, 1, 2, 3, 4]
            assert self.batches["val"]["to_tensor_transform"] == [0, 1, 2, 3, 4]
            assert self.batches["val"]["post_tensor_transform"] == [0, 1, 2, 3, 4]
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

            preprocess = cls.preprocess_cls()

            return cls.from_load_data_inputs(
                train_load_data_input=train_data,
                val_load_data_input=val_data,
                test_load_data_input=test_data,
                predict_load_data_input=predict_data,
                preprocess=preprocess,
                batch_size=5
            )

    dm = CustomDataModule.from_inputs(range(5), range(5), range(5), range(5))
    data_fetcher: CheckData = dm.data_fetcher

    with data_fetcher.enable():
        _ = next(iter(dm.val_dataloader()))

    data_fetcher.check()
    data_fetcher.reset()
    assert data_fetcher.batches == {'train': {}, 'test': {}, 'val': {}, 'predict': {}}


def test_base_viz(tmpdir):

    seed_everything(42)
    tmpdir = Path(tmpdir)

    (tmpdir / "a").mkdir()
    (tmpdir / "b").mkdir()
    _rand_image().save(tmpdir / "a" / "a_1.png")
    _rand_image().save(tmpdir / "b" / "a_1.png")

    class CustomBaseVisualization(BaseVisualization):

        show_load_sample_called = False
        show_pre_tensor_transform_called = False
        show_to_tensor_transform_called = False
        show_post_tensor_transform_called = False
        show_collate_called = False
        per_batch_transform_called = False

        def show_load_sample(self, samples: List[Any], running_stage: RunningStage):
            self.show_load_sample_called = True

        def show_pre_tensor_transform(self, samples: List[Any], running_stage: RunningStage):
            self.show_pre_tensor_transform_called = True

        def show_to_tensor_transform(self, samples: List[Any], running_stage: RunningStage):
            self.show_to_tensor_transform_called = True

        def show_post_tensor_transform(self, samples: List[Any], running_stage: RunningStage):
            self.show_post_tensor_transform_called = True

        def show_collate(self, batch: Sequence, running_stage: RunningStage) -> None:
            self.show_collate_called = True

        def show_per_batch_transform(self, batch: Sequence, running_stage: RunningStage) -> None:
            self.per_batch_transform_called = True

        def check_reset(self):
            self.show_load_sample_called = False
            self.show_pre_tensor_transform_called = False
            self.show_to_tensor_transform_called = False
            self.show_post_tensor_transform_called = False
            self.show_collate_called = False
            self.per_batch_transform_called = False

    class CustomImageClassificationData(ImageClassificationData):

        @staticmethod
        def configure_data_fetcher(*args, **kwargs) -> CustomBaseVisualization:
            return CustomBaseVisualization(*args, **kwargs)

    dm = CustomImageClassificationData.from_filepaths(
        train_filepaths=[tmpdir / "a", tmpdir / "b"],
        train_labels=[0, 1],
        val_filepaths=[tmpdir / "a", tmpdir / "b"],
        val_labels=[0, 1],
        test_filepaths=[tmpdir / "a", tmpdir / "b"],
        test_labels=[0, 1],
        predict_filepaths=[tmpdir / "a", tmpdir / "b"],
        batch_size=2,
        num_workers=0,
    )

    for stage in _STAGES_PREFIX.values():

        for _ in range(10):
            getattr(dm, f"show_{stage}_batch")(reset=False)

        is_predict = stage == "predict"

        def extract_data(data):
            if not is_predict:
                return data[0][0]
            return data[0]

        assert isinstance(extract_data(dm.data_fetcher.batches[stage]["load_sample"]), Image.Image)
        if not is_predict:
            assert isinstance(dm.data_fetcher.batches[stage]["load_sample"][0][1], int)

        assert isinstance(extract_data(dm.data_fetcher.batches[stage]["to_tensor_transform"]), torch.Tensor)
        if not is_predict:
            assert isinstance(dm.data_fetcher.batches[stage]["to_tensor_transform"][0][1], int)

        assert extract_data(dm.data_fetcher.batches[stage]["collate"]).shape == torch.Size([2, 3, 196, 196])
        if not is_predict:
            assert dm.data_fetcher.batches[stage]["collate"][0][1].shape == torch.Size([2])

        generated = extract_data(dm.data_fetcher.batches[stage]["per_batch_transform"]).shape
        assert generated == torch.Size([2, 3, 196, 196])
        if not is_predict:
            assert dm.data_fetcher.batches[stage]["per_batch_transform"][0][1].shape == torch.Size([2])

        assert dm.data_fetcher.show_load_sample_called
        assert dm.data_fetcher.show_pre_tensor_transform_called
        assert dm.data_fetcher.show_to_tensor_transform_called
        assert dm.data_fetcher.show_post_tensor_transform_called
        assert dm.data_fetcher.show_collate_called
        assert dm.data_fetcher.per_batch_transform_called
        dm.data_fetcher.check_reset()


def test_data_loaders_num_workers_to_0(tmpdir):
    """
    num_workers should be set to `0` internally for visualization and not for training.
    """

    datamodule = DataModule(train_dataset=range(10), num_workers=3)
    iterator = datamodule._reset_iterator(RunningStage.TRAINING)
    assert isinstance(iterator, torch.utils.data.dataloader._SingleProcessDataLoaderIter)
    iterator = iter(datamodule.train_dataloader())
    assert isinstance(iterator, torch.utils.data.dataloader._MultiProcessingDataLoaderIter)
    assert datamodule.num_workers == 3
