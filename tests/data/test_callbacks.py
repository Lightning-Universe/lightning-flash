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
from flash.data.data_source import DefaultDataKeys
from flash.data.process import DefaultPreprocess
from flash.data.utils import _CALLBACK_FUNCS, _STAGES_PREFIX
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

            preprocess = DefaultPreprocess()

            return cls.from_data_source(
                "default",
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                predict_data=predict_data,
                preprocess=preprocess,
                batch_size=5,
            )

    dm = CustomDataModule.from_inputs(range(5), range(5), range(5), range(5))
    data_fetcher: CheckData = dm.data_fetcher

    with data_fetcher.enable():
        _ = next(iter(dm.val_dataloader()))

    # TODO: the method below fails because the data fetcher internally doesn't seem to cache
    # properly the batches at each stage.
    data_fetcher.check()
    data_fetcher.reset()
    assert data_fetcher.batches == {'train': {}, 'test': {}, 'val': {}, 'predict': {}}


def test_base_viz(tmpdir):

    seed_everything(42)
    tmpdir = Path(tmpdir)

    train_images = [str(tmpdir / "a1.png"), str(tmpdir / "b1.png")]

    _rand_image().save(train_images[0])
    _rand_image().save(train_images[1])

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

    B: int = 2  # batch_size

    dm = CustomImageClassificationData.from_files(
        train_files=train_images,
        train_targets=[0, 1],
        val_files=train_images,
        val_targets=[2, 3],
        test_files=train_images,
        test_targets=[4, 5],
        predict_files=train_images,
        batch_size=B,
        num_workers=0,
    )

    num_tests = 10

    for stage in _STAGES_PREFIX.values():

        for _ in range(num_tests):
            for fcn_name in _CALLBACK_FUNCS:
                fcn = getattr(dm, f"show_{stage}_batch")
                fcn(fcn_name, reset=True)

        is_predict = stage == "predict"

        def _extract_data(data):
            return data[0][DefaultDataKeys.INPUT]

        def _get_result(function_name: str):
            return dm.data_fetcher.batches[stage][function_name]

        res = _get_result("load_sample")
        assert len(res) == B
        assert isinstance(_extract_data(res), Image.Image)

        if not is_predict:
            res = _get_result("load_sample")
            assert isinstance(res[0][DefaultDataKeys.TARGET], int)

        res = _get_result("to_tensor_transform")
        assert len(res) == B
        assert isinstance(_extract_data(res), torch.Tensor)

        if not is_predict:
            res = _get_result("to_tensor_transform")
            assert isinstance(res[0][DefaultDataKeys.TARGET], torch.Tensor)

        res = _get_result("collate")
        assert _extract_data(res).shape == (B, 3, 196, 196)

        if not is_predict:
            res = _get_result("collate")
            assert res[0][DefaultDataKeys.TARGET].shape == torch.Size([2])

        res = _get_result("per_batch_transform")
        assert _extract_data(res).shape == (B, 3, 196, 196)

        if not is_predict:
            res = _get_result("per_batch_transform")
            assert res[0][DefaultDataKeys.TARGET].shape == (B, )

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
