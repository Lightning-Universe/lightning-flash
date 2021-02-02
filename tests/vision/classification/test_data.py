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

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from flash.vision import ImageClassificationData


def _dummy_image_loader(filepath):
    return torch.rand(3, 64, 64)


def _rand_image():
    return Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))


def test_from_filepaths(tmpdir):
    img_data = ImageClassificationData.from_filepaths(
        train_filepaths=["a", "b"],
        train_labels=[0, 1],
        train_transform=lambda x: x,  # make sure transform works
        loader=_dummy_image_loader,
        batch_size=1,
        num_workers=0,
    )

    data = next(iter(img_data.train_dataloader()))
    imgs, labels = data
    assert imgs.shape == (1, 3, 64, 64)
    assert labels.shape == (1, )

    assert img_data.val_dataloader() is None
    assert img_data.test_dataloader() is None

    img_data = ImageClassificationData.from_filepaths(
        train_filepaths=["a", "b"],
        train_labels=[0, 1],
        train_transform=None,
        valid_filepaths=["c", "d"],
        valid_labels=[0, 1],
        valid_transform=None,
        test_filepaths=["e", "f"],
        test_labels=[0, 1],
        loader=_dummy_image_loader,
        batch_size=1,
        num_workers=0,
    )

    data = next(iter(img_data.val_dataloader()))
    imgs, labels = data
    assert imgs.shape == (1, 3, 64, 64)
    assert labels.shape == (1, )

    data = next(iter(img_data.test_dataloader()))
    imgs, labels = data
    assert imgs.shape == (1, 3, 64, 64)
    assert labels.shape == (1, )


def test_from_folders(tmpdir):
    train_dir = Path(tmpdir / "train")
    train_dir.mkdir()

    (train_dir / "a").mkdir()
    _rand_image().save(train_dir / "a" / "1.png")
    _rand_image().save(train_dir / "a" / "2.png")

    (train_dir / "b").mkdir()
    _rand_image().save(train_dir / "b" / "1.png")
    _rand_image().save(train_dir / "b" / "2.png")

    img_data = ImageClassificationData.from_folders(
        train_dir, train_transform=None, loader=_dummy_image_loader, batch_size=1
    )
    data = next(iter(img_data.train_dataloader()))
    imgs, labels = data
    assert imgs.shape == (1, 3, 64, 64)
    assert labels.shape == (1, )

    assert img_data.val_dataloader() is None
    assert img_data.test_dataloader() is None

    img_data = ImageClassificationData.from_folders(
        train_dir,
        train_transform=T.ToTensor(),
        valid_folder=train_dir,
        valid_transform=T.ToTensor(),
        test_folder=train_dir,
        batch_size=1,
        num_workers=0,
    )

    data = next(iter(img_data.val_dataloader()))
    imgs, labels = data
    assert imgs.shape == (1, 3, 64, 64)
    assert labels.shape == (1, )

    data = next(iter(img_data.test_dataloader()))
    imgs, labels = data
    assert imgs.shape == (1, 3, 64, 64)
    assert labels.shape == (1, )
