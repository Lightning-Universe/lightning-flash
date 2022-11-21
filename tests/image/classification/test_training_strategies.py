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
import os
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from flash import Trainer
from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _IMAGE_TESTING, _LEARN2LEARN_AVAILABLE
from flash.image import ImageClassificationData, ImageClassifier
from flash.image.classification.adapters import TRAINING_STRATEGIES
from tests.image.classification.test_data import _rand_image

# ======== Mock functions ========


class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        return {
            DataKeys.INPUT: torch.rand(3, 96, 96),
            DataKeys.TARGET: torch.randint(10, size=(1,)).item(),
        }

    def __len__(self) -> int:
        return 2


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_default_strategies(tmpdir):
    num_classes = 10
    ds = DummyDataset()
    model = ImageClassifier(num_classes, backbone="resnet50")

    trainer = Trainer(fast_dev_run=2)
    trainer.fit(model, train_dataloader=DataLoader(ds))


@pytest.mark.skipif(not _LEARN2LEARN_AVAILABLE, reason="image and learn2learn libraries aren't installed.")
def test_learn2learn_training_strategies_registry():
    assert TRAINING_STRATEGIES.available_keys() == ["anil", "default", "maml", "metaoptnet", "prototypicalnetworks"]


def _test_learn2learning_training_strategies(gpus, training_strategy, tmpdir, accelerator=None, strategy=None):
    train_dir = Path(tmpdir / "train")
    train_dir.mkdir()

    (train_dir / "a").mkdir()
    pa_1 = train_dir / "a" / "1.png"
    pa_2 = train_dir / "a" / "2.png"
    pb_1 = train_dir / "b" / "1.png"
    pb_2 = train_dir / "b" / "2.png"
    image_size = (96, 96)
    _rand_image(image_size).save(pa_1)
    _rand_image(image_size).save(pa_2)

    (train_dir / "b").mkdir()
    _rand_image(image_size).save(pb_1)
    _rand_image(image_size).save(pb_2)

    n = 5

    dm = ImageClassificationData.from_files(
        train_files=[str(pa_1)] * n + [str(pa_2)] * n + [str(pb_1)] * n + [str(pb_2)] * n,
        train_targets=[0] * n + [1] * n + [2] * n + [3] * n,
        batch_size=1,
        num_workers=0,
        transform_kwargs=dict(image_size=image_size),
    )

    model = ImageClassifier(
        backbone="resnet18",
        training_strategy=training_strategy,
        training_strategy_kwargs={"ways": dm.num_classes, "shots": 4, "meta_batch_size": 4},
    )

    trainer = Trainer(fast_dev_run=2, gpus=gpus, strategy=strategy)

    trainer.fit(model, datamodule=dm)


# 'metaoptnet' is not yet supported as it requires qpth as a dependency.
@pytest.mark.parametrize("training_strategy", ["anil", "maml", "prototypicalnetworks"])
@pytest.mark.skipif(not _LEARN2LEARN_AVAILABLE, reason="image and learn2learn libraries aren't installed.")
def test_learn2learn_training_strategies(training_strategy, tmpdir):
    _test_learn2learning_training_strategies(0, training_strategy, tmpdir, accelerator=None)


@pytest.mark.skipif(not _LEARN2LEARN_AVAILABLE, reason="image and learn2learn libraries aren't installed.")
def test_wrongly_specified_training_strategies():
    with pytest.raises(KeyError, match="something is not in FlashRegistry"):
        ImageClassifier(
            backbone="resnet18",
            training_strategy="something",
            training_strategy_kwargs={"ways": 2, "shots": 4, "meta_batch_size": 10},
        )


@pytest.mark.skipif(not os.getenv("FLASH_RUNNING_SPECIAL_TESTS", "0") == "1", reason="Should run with special test")
@pytest.mark.skipif(not _LEARN2LEARN_AVAILABLE, reason="image and learn2learn libraries aren't installed.")
def test_learn2learn_training_strategies_ddp(tmpdir):
    _test_learn2learning_training_strategies(2, "prototypicalnetworks", tmpdir, strategy="ddp")
