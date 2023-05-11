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
from typing import Any, List, Sequence, Tuple

import numpy as np
import pytest
import torch
from pytorch_lightning import seed_everything
from torch import Tensor

from flash.core.data.base_viz import BaseVisualization
from flash.core.data.io.input import DataKeys
from flash.core.data.utils import _CALLBACK_FUNCS
from flash.core.utilities.imports import _PIL_AVAILABLE, _TOPIC_IMAGE_AVAILABLE
from flash.core.utilities.stages import RunningStage
from flash.image import ImageClassificationData

if _PIL_AVAILABLE:
    from PIL import Image


def _rand_image():
    return Image.fromarray(np.random.randint(0, 255, (196, 196, 3), dtype="uint8"))


class CustomBaseVisualization(BaseVisualization):
    def __init__(self):
        super().__init__()
        self.check_reset()

    def show_load_sample(
        self,
        samples: List[Any],
        running_stage: RunningStage,
        limit_nb_samples: int = None,
        figsize: Tuple[int, int] = (6.4, 4.8),
    ):
        self.show_load_sample_called = True

    def show_per_sample_transform(
        self,
        samples: List[Any],
        running_stage: RunningStage,
        limit_nb_samples: int = None,
        figsize: Tuple[int, int] = (6.4, 4.8),
    ):
        self.show_per_sample_transform_called = True

    def show_collate(
        self,
        batch: Sequence,
        running_stage: RunningStage,
        limit_nb_samples: int = None,
        figsize: Tuple[int, int] = (6.4, 4.8),
    ) -> None:
        self.show_collate_called = True

    def show_per_batch_transform(
        self,
        batch: Sequence,
        running_stage: RunningStage,
        limit_nb_samples: int = None,
        figsize: Tuple[int, int] = (6.4, 4.8),
    ) -> None:
        self.per_batch_transform_called = True

    def check_reset(self):
        self.show_load_sample_called = False
        self.show_per_sample_transform_called = False
        self.show_collate_called = False
        self.per_batch_transform_called = False


@pytest.mark.skipif(not _TOPIC_IMAGE_AVAILABLE, reason="image libraries aren't installed.")
class TestBaseViz:
    def test_base_viz(self, tmpdir):
        seed_everything(42)
        tmpdir = Path(tmpdir)

        train_images = [str(tmpdir / "a1.png"), str(tmpdir / "b1.png")]

        _rand_image().save(train_images[0])
        _rand_image().save(train_images[1])

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

        for stage in ("train", "val", "test", "predict"):
            for _ in range(num_tests):
                for fcn_name in _CALLBACK_FUNCS:
                    dm.data_fetcher.reset()
                    assert dm.data_fetcher.batches == {"predict": {}, "test": {}, "train": {}, "val": {}, "serve": {}}
                    fcn = getattr(dm, f"show_{stage}_batch")
                    fcn(fcn_name, reset=False)

            is_predict = stage == "predict"

            def _extract_data(data):
                return data[0][DataKeys.INPUT]

            def _get_result(function_name: str):
                return dm.data_fetcher.batches[stage][function_name]

            res = _get_result("load_sample")
            assert len(res) == B
            assert isinstance(_extract_data(res), Image.Image)

            if not is_predict:
                res = _get_result("load_sample")
                assert isinstance(res[0][DataKeys.TARGET], int)

            res = _get_result("per_sample_transform")
            assert len(res) == B
            assert isinstance(_extract_data(res), Tensor)

            if not is_predict:
                res = _get_result("per_sample_transform")
                assert isinstance(res[0][DataKeys.TARGET], Tensor)

            res = _get_result("collate")
            assert _extract_data(res).shape == (B, 3, 196, 196)

            if not is_predict:
                res = _get_result("collate")
                assert res[0][DataKeys.TARGET].shape == torch.Size([2])

            res = _get_result("per_batch_transform")
            assert _extract_data(res).shape == (B, 3, 196, 196)

            if not is_predict:
                res = _get_result("per_batch_transform")
                assert res[0][DataKeys.TARGET].shape == (B,)

            assert dm.data_fetcher.show_load_sample_called
            assert dm.data_fetcher.show_per_sample_transform_called
            assert dm.data_fetcher.show_collate_called
            assert dm.data_fetcher.per_batch_transform_called
            dm.data_fetcher.reset()

    @pytest.mark.parametrize(
        "func_names, valid",
        [
            (["load_sample"], True),
            (["not_a_hook"], False),
            (["load_sample", "per_sample_transform"], True),
            (["load_sample", "not_a_hook"], True),
        ],
    )
    def test_show(self, func_names, valid):
        base_viz = CustomBaseVisualization()

        batch = {func_name: "test" for func_name in func_names}

        if not valid:
            with pytest.raises(ValueError, match="Invalid function names"):
                base_viz.show(batch, RunningStage.TRAINING, func_names)
        else:
            base_viz.show(batch, RunningStage.TRAINING, func_names)
            for func_name in func_names:
                if hasattr(base_viz, f"show_{func_name}_called"):
                    assert getattr(base_viz, f"show_{func_name}_called")
