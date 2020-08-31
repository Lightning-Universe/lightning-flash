from pl_flash.vision import ImageClassificationData
from pathlib import Path

import torch
import pytest


def _dummy_image_loader(filepath):
    return torch.rand(3, 64, 64)


def test_from_filepaths(tmpdir):
    img_data = ImageClassificationData.from_filepaths(
        train_filepaths=["a", "b"], train_labels=[0, 1], loader=_dummy_image_loader, batch_size=1,
    )

    imgs, labels = next(iter(img_data.train_dataloader()))
    assert imgs.shape == (1, 3, 64, 64)
    assert labels.shape == (1,)

    assert img_data.val_dataloader() is None
    assert img_data.test_dataloader() is None

    img_data = ImageClassificationData.from_filepaths(
        train_filepaths=["a", "b"],
        train_labels=[0, 1],
        valid_filepaths=["c", "d"],
        valid_labels=[0, 1],
        test_filepaths=["e", "f"],
        test_labels=[0, 1],
        loader=_dummy_image_loader,
        batch_size=1,
    )

    imgs, labels = next(iter(img_data.val_dataloader()))
    assert imgs.shape == (1, 3, 64, 64)
    assert labels.shape == (1,)

    imgs, labels = next(iter(img_data.test_dataloader()))
    assert imgs.shape == (1, 3, 64, 64)
    assert labels.shape == (1,)


def test_from_folders(tmpdir):
    train_dir = Path(tmpdir / "train")
    train_dir.mkdir()

    (train_dir / "a").mkdir()
    Path(train_dir / "a" / "1.png").touch()
    Path(train_dir / "a" / "2.png").touch()

    (train_dir / "b").mkdir()
    Path(train_dir / "b" / "1.png").touch()
    Path(train_dir / "b" / "2.png").touch()

    img_data = ImageClassificationData.from_folders(train_dir, loader=_dummy_image_loader, batch_size=1)
    imgs, labels = next(iter(img_data.train_dataloader()))
    assert imgs.shape == (1, 3, 64, 64)
    assert labels.shape == (1,)

    assert img_data.val_dataloader() is None
    assert img_data.test_dataloader() is None

    img_data = ImageClassificationData.from_folders(
        train_dir, valid_folder=train_dir, test_folder=train_dir, loader=_dummy_image_loader, batch_size=1,
    )

    imgs, labels = next(iter(img_data.val_dataloader()))
    assert imgs.shape == (1, 3, 64, 64)
    assert labels.shape == (1,)

    imgs, labels = next(iter(img_data.test_dataloader()))
    assert imgs.shape == (1, 3, 64, 64)
    assert labels.shape == (1,)
