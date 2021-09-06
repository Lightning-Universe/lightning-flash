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

import pytest
import torch
from torch.utils.data import DataLoader

from flash import Trainer
from flash.core.data.data_source import DefaultDataKeys
from flash.core.utilities.imports import _LEARN2LEARN_AVAILABLE
from flash.image import ImageClassificationData, ImageClassifier
from tests.image.classification.test_data import _rand_image

# ======== Mock functions ========


class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        return {
            DefaultDataKeys.INPUT: torch.rand(3, 224, 224),
            DefaultDataKeys.TARGET: torch.randint(10, size=(1,)).item(),
        }

    def __len__(self) -> int:
        return 100


@pytest.mark.skipif(not _LEARN2LEARN_AVAILABLE, reason="image and learn2learn libraries aren't installed.")
def test_learn2learn_strategies(tmpdir):
    ds = DummyDataset()
    model = ImageClassifier(10, backbone="resnet50", training_strategy="default")

    trainer = Trainer(fast_dev_run=2)
    trainer.fit(model, train_dataloader=DataLoader(ds))

    model = ImageClassifier(
        10,
        backbone="resnet50",
        training_strategy="maml",
        training_strategy_kwargs={"train_samples": 4, "train_ways": 4, "test_samples": 10, "test_ways": 4},
    )

    train_dir = Path(tmpdir / "train")
    train_dir.mkdir()

    (train_dir / "a").mkdir()
    pa_1 = train_dir / "a" / "1.png"
    pa_2 = train_dir / "a" / "2.png"
    pb_1 = train_dir / "b" / "1.png"
    pb_2 = train_dir / "b" / "2.png"
    _rand_image().save(pa_1)
    _rand_image().save(pa_2)

    (train_dir / "b").mkdir()
    _rand_image().save(pb_1)
    _rand_image().save(pb_2)

    dm = ImageClassificationData.from_files(
        train_files=[str(pa_1)] * 5 + [str(pa_2)] * 5 + [str(pb_1)] * 5 + [str(pb_2)] * 5,
        train_targets=[0] * 5 + [1] * 5 + [2] * 5 + [3] * 5,
        batch_size=1,
        num_workers=0,
    )

    trainer = Trainer(fast_dev_run=2)
    trainer.fit(model, datamodule=dm)
