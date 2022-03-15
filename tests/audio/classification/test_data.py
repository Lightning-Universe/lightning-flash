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
from typing import Any, List, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
from pytorch_lightning import seed_everything

from flash.audio import AudioClassificationData
from flash.core.data.io.input import DataKeys
from flash.core.data.transforms import ApplyToKeys
from flash.core.utilities.imports import _AUDIO_TESTING, _MATPLOTLIB_AVAILABLE, _PIL_AVAILABLE, _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    import torchvision.transforms as T

if _PIL_AVAILABLE:
    from PIL import Image


def _rand_image(size: Tuple[int, int] = None):
    if size is None:
        _size = np.random.choice([196, 244])
        size = (_size, _size)
    return Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype="uint8"))


@pytest.mark.skipif(not _AUDIO_TESTING, reason="audio libraries aren't installed.")
def test_from_filepaths_smoke(tmpdir):

    tmpdir = Path(tmpdir)

    _rand_image().save(tmpdir / "a_1.png")
    _rand_image().save(tmpdir / "b_1.png")

    train_images = [
        str(tmpdir / "a_1.png"),
        str(tmpdir / "b_1.png"),
    ]

    spectrograms_data = AudioClassificationData.from_files(
        train_files=train_images,
        train_targets=[1, 2],
        batch_size=2,
        num_workers=0,
    )
    assert spectrograms_data.train_dataloader() is not None

    data = next(iter(spectrograms_data.train_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (2, 3, 128, 128)
    assert labels.shape == (2,)
    assert sorted(list(labels.numpy())) == [1, 2]


@pytest.mark.skipif(not _AUDIO_TESTING, reason="audio libraries aren't installed.")
@pytest.mark.parametrize(
    "data,from_function",
    [
        (torch.rand(3, 3, 64, 64), AudioClassificationData.from_tensors),
        (np.random.rand(3, 3, 64, 64), AudioClassificationData.from_numpy),
    ],
)
def test_from_data(data, from_function):
    img_data = from_function(
        train_data=data,
        train_targets=[0, 3, 6],
        val_data=data,
        val_targets=[1, 4, 7],
        test_data=data,
        test_targets=[2, 5, 8],
        batch_size=2,
        num_workers=0,
    )

    # check training data
    data = next(iter(img_data.train_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (2, 3, 128, 128)
    assert labels.shape == (2,)
    assert labels.numpy()[0] in [0, 3, 6]  # data comes shuffled here
    assert labels.numpy()[1] in [0, 3, 6]  # data comes shuffled here

    # check validation data
    data = next(iter(img_data.val_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (2, 3, 128, 128)
    assert labels.shape == (2,)
    assert list(labels.numpy()) == [1, 4]

    # check test data
    data = next(iter(img_data.test_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (2, 3, 128, 128)
    assert labels.shape == (2,)
    assert list(labels.numpy()) == [2, 5]


@pytest.mark.skipif(not _AUDIO_TESTING, reason="audio libraries aren't installed.")
def test_from_filepaths_numpy(tmpdir):
    tmpdir = Path(tmpdir)

    np.save(str(tmpdir / "a_1.npy"), np.random.rand(64, 64, 3))
    np.save(str(tmpdir / "b_1.npy"), np.random.rand(64, 64, 3))

    train_images = [
        str(tmpdir / "a_1.npy"),
        str(tmpdir / "b_1.npy"),
    ]

    spectrograms_data = AudioClassificationData.from_files(
        train_files=train_images,
        train_targets=[1, 2],
        batch_size=2,
        num_workers=0,
    )
    assert spectrograms_data.train_dataloader() is not None

    data = next(iter(spectrograms_data.train_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (2, 3, 128, 128)
    assert labels.shape == (2,)
    assert sorted(list(labels.numpy())) == [1, 2]


@pytest.mark.skipif(not _AUDIO_TESTING, reason="audio libraries aren't installed.")
def test_from_filepaths_list_image_paths(tmpdir):
    tmpdir = Path(tmpdir)

    _rand_image().save(tmpdir / "e_1.png")

    train_images = [
        str(tmpdir / "e_1.png"),
        str(tmpdir / "e_1.png"),
        str(tmpdir / "e_1.png"),
    ]

    spectrograms_data = AudioClassificationData.from_files(
        train_files=train_images,
        train_targets=[0, 3, 6],
        val_files=train_images,
        val_targets=[1, 4, 7],
        test_files=train_images,
        test_targets=[2, 5, 8],
        batch_size=2,
        num_workers=0,
    )

    # check training data
    data = next(iter(spectrograms_data.train_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (2, 3, 128, 128)
    assert labels.shape == (2,)
    assert labels.numpy()[0] in [0, 3, 6]  # data comes shuffled here
    assert labels.numpy()[1] in [0, 3, 6]  # data comes shuffled here

    # check validation data
    data = next(iter(spectrograms_data.val_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (2, 3, 128, 128)
    assert labels.shape == (2,)
    assert list(labels.numpy()) == [1, 4]

    # check test data
    data = next(iter(spectrograms_data.test_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (2, 3, 128, 128)
    assert labels.shape == (2,)
    assert list(labels.numpy()) == [2, 5]


@pytest.mark.skipif(not _AUDIO_TESTING, reason="audio libraries aren't installed.")
@pytest.mark.skipif(not _MATPLOTLIB_AVAILABLE, reason="matplotlib isn't installed.")
def test_from_filepaths_visualise(tmpdir):
    tmpdir = Path(tmpdir)

    _rand_image().save(tmpdir / "e_1.png")

    train_images = [
        str(tmpdir / "e_1.png"),
        str(tmpdir / "e_1.png"),
        str(tmpdir / "e_1.png"),
    ]

    dm = AudioClassificationData.from_files(
        train_files=train_images,
        train_targets=[0, 3, 6],
        val_files=train_images,
        val_targets=[1, 4, 7],
        test_files=train_images,
        test_targets=[2, 5, 8],
        batch_size=2,
        num_workers=0,
    )

    # disable visualisation for testing
    assert dm.data_fetcher.block_viz_window is True
    dm.set_block_viz_window(False)
    assert dm.data_fetcher.block_viz_window is False

    # call show functions
    # dm.show_train_batch()
    dm.show_train_batch("per_sample_transform")
    dm.show_train_batch(["per_sample_transform", "per_batch_transform"])


@pytest.mark.skipif(not _AUDIO_TESTING, reason="audio libraries aren't installed.")
@pytest.mark.skipif(not _MATPLOTLIB_AVAILABLE, reason="matplotlib isn't installed.")
def test_from_filepaths_visualise_multilabel(tmpdir):
    tmpdir = Path(tmpdir)

    (tmpdir / "a").mkdir()
    (tmpdir / "b").mkdir()

    image_a = str(tmpdir / "a" / "a_1.png")
    image_b = str(tmpdir / "b" / "b_1.png")

    _rand_image().save(image_a)
    _rand_image().save(image_b)

    dm = AudioClassificationData.from_files(
        train_files=[image_a, image_b],
        train_targets=[[0, 1, 0], [0, 1, 1]],
        val_files=[image_b, image_a],
        val_targets=[[1, 1, 0], [0, 0, 1]],
        test_files=[image_b, image_b],
        test_targets=[[0, 0, 1], [1, 1, 0]],
        batch_size=2,
        transform_kwargs=dict(spectrogram_size=(64, 64)),
    )
    # disable visualisation for testing
    assert dm.data_fetcher.block_viz_window is True
    dm.set_block_viz_window(False)
    assert dm.data_fetcher.block_viz_window is False

    # call show functions
    dm.show_train_batch()
    dm.show_train_batch("per_sample_transform")
    dm.show_val_batch("per_batch_transform")


@pytest.mark.skipif(not _AUDIO_TESTING, reason="audio libraries aren't installed.")
def test_from_filepaths_splits(tmpdir):
    tmpdir = Path(tmpdir)

    B, _, H, W = 2, 3, 224, 224
    img_size: Tuple[int, int] = (H, W)

    (tmpdir / "splits").mkdir()
    _rand_image(img_size).save(tmpdir / "s.png")

    num_samples: int = 10
    val_split: float = 0.3

    train_filepaths: List[str] = [str(tmpdir / "s.png") for _ in range(num_samples)]

    train_labels: List[int] = list(range(num_samples))

    assert len(train_filepaths) == len(train_labels)

    _to_tensor = nn.Sequential(
        ApplyToKeys(DataKeys.INPUT, T.Compose([T.ToTensor(), T.Resize(img_size)])),
        ApplyToKeys(DataKeys.TARGET, torch.as_tensor),
    )

    def run(transform: Any = None):
        dm = AudioClassificationData.from_files(
            train_files=train_filepaths,
            train_targets=train_labels,
            transform=transform,
            batch_size=B,
            num_workers=0,
            val_split=val_split,
        )
        data = next(iter(dm.train_dataloader()))
        imgs, labels = data["input"], data["target"]
        assert imgs.shape == (B, 3, H, W)
        assert labels.shape == (B,)

    run(_to_tensor)


@pytest.mark.skipif(not _AUDIO_TESTING, reason="audio libraries aren't installed.")
def test_from_folders_only_train(tmpdir):

    seed_everything(42)

    train_dir = Path(tmpdir / "train")
    train_dir.mkdir()

    (train_dir / "a").mkdir()
    _rand_image().save(train_dir / "a" / "1.png")
    _rand_image().save(train_dir / "a" / "2.png")

    (train_dir / "b").mkdir()
    _rand_image().save(train_dir / "b" / "1.png")
    _rand_image().save(train_dir / "b" / "2.png")

    spectrograms_data = AudioClassificationData.from_folders(train_dir, transform=None, batch_size=1)

    data = next(iter(spectrograms_data.train_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (1, 196, 196, 3)
    assert labels.shape == (1,)


@pytest.mark.skipif(not _AUDIO_TESTING, reason="audio libraries aren't installed.")
def test_from_folders_train_val(tmpdir):
    seed_everything(42)

    train_dir = Path(tmpdir / "train")
    train_dir.mkdir()

    (train_dir / "a").mkdir()
    _rand_image().save(train_dir / "a" / "1.png")
    _rand_image().save(train_dir / "a" / "2.png")

    (train_dir / "b").mkdir()
    _rand_image().save(train_dir / "b" / "1.png")
    _rand_image().save(train_dir / "b" / "2.png")
    spectrograms_data = AudioClassificationData.from_folders(
        train_dir,
        val_folder=train_dir,
        test_folder=train_dir,
        batch_size=2,
        num_workers=0,
    )

    data = next(iter(spectrograms_data.train_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (2, 3, 128, 128)
    assert labels.shape == (2,)
    assert list(labels.numpy()) == [0, 1]

    data = next(iter(spectrograms_data.val_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (2, 3, 128, 128)
    assert labels.shape == (2,)
    assert list(labels.numpy()) == [0, 0]

    data = next(iter(spectrograms_data.test_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (2, 3, 128, 128)
    assert labels.shape == (2,)
    assert list(labels.numpy()) == [0, 0]


@pytest.mark.skipif(not _AUDIO_TESTING, reason="audio libraries aren't installed.")
def test_from_filepaths_multilabel(tmpdir):
    tmpdir = Path(tmpdir)

    (tmpdir / "a").mkdir()
    _rand_image().save(tmpdir / "a1.png")
    _rand_image().save(tmpdir / "a2.png")

    train_images = [str(tmpdir / "a1.png"), str(tmpdir / "a2.png")]
    train_labels = [[1, 0, 1, 0], [0, 0, 1, 1]]
    valid_labels = [[1, 1, 1, 0], [1, 0, 0, 1]]
    test_labels = [[1, 0, 1, 0], [1, 1, 0, 1]]

    dm = AudioClassificationData.from_files(
        train_files=train_images,
        train_targets=train_labels,
        val_files=train_images,
        val_targets=valid_labels,
        test_files=train_images,
        test_targets=test_labels,
        batch_size=2,
        num_workers=0,
    )

    data = next(iter(dm.train_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (2, 3, 128, 128)
    assert labels.shape == (2, 4)

    data = next(iter(dm.val_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (2, 3, 128, 128)
    assert labels.shape == (2, 4)
    torch.testing.assert_allclose(labels, torch.tensor(valid_labels))

    data = next(iter(dm.test_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (2, 3, 128, 128)
    assert labels.shape == (2, 4)
    torch.testing.assert_allclose(labels, torch.tensor(test_labels))
