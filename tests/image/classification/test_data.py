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
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import torch

from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import (
    _FIFTYONE_AVAILABLE,
    _IMAGE_AVAILABLE,
    _IMAGE_TESTING,
    _MATPLOTLIB_AVAILABLE,
    _PIL_AVAILABLE,
    _TORCHVISION_AVAILABLE,
)
from flash.image import ImageClassificationData, ImageClassificationInputTransform

if _TORCHVISION_AVAILABLE:
    from torchvision.datasets import FakeData

if _PIL_AVAILABLE:
    from PIL import Image

if _FIFTYONE_AVAILABLE:
    import fiftyone as fo


def _rand_image(size: Tuple[int, int] = None):
    if size is None:
        _size = np.random.choice([196, 244])
        size = (_size, _size)
    return Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype="uint8"))


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_from_filepaths_smoke(tmpdir):
    tmpdir = Path(tmpdir)

    (tmpdir / "a").mkdir()
    (tmpdir / "b").mkdir()
    _rand_image().save(tmpdir / "a_1.png")
    _rand_image().save(tmpdir / "b_1.png")

    train_images = [
        str(tmpdir / "a_1.png"),
        str(tmpdir / "b_1.png"),
    ]

    img_data = ImageClassificationData.from_files(
        train_files=train_images,
        train_targets=[1, 2],
        batch_size=2,
        num_workers=0,
    )
    assert img_data.train_dataloader() is not None

    data = next(iter(img_data.train_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (2, 3, 196, 196)
    assert labels.shape == (2,)
    assert sorted(list(labels.numpy())) == [1, 2]


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_from_data_frame_smoke(tmpdir):
    tmpdir = Path(tmpdir)

    df = pd.DataFrame(
        {"file": ["train.png", "valid.png", "test.png"], "split": ["train", "valid", "test"], "target": [0, 1, 1]}
    )

    [_rand_image().save(tmpdir / row.file) for i, row in df.iterrows()]

    img_data = ImageClassificationData.from_data_frame(
        "file",
        "target",
        train_images_root=str(tmpdir),
        val_images_root=str(tmpdir),
        test_images_root=str(tmpdir),
        train_data_frame=df[df.split == "train"],
        val_data_frame=df[df.split == "valid"],
        test_data_frame=df[df.split == "test"],
        predict_images_root=str(tmpdir),
        batch_size=1,
        predict_data_frame=df,
    )

    assert img_data.train_dataloader() is not None
    assert img_data.val_dataloader() is not None
    assert img_data.test_dataloader() is not None
    assert img_data.predict_dataloader() is not None

    data = next(iter(img_data.train_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (1, 3, 196, 196)
    assert labels.shape == (1,)
    assert sorted(list(labels.numpy())) == [0]

    data = next(iter(img_data.val_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (1, 3, 196, 196)
    assert labels.shape == (1,)
    assert sorted(list(labels.numpy())) == [1]

    data = next(iter(img_data.test_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (1, 3, 196, 196)
    assert labels.shape == (1,)
    assert sorted(list(labels.numpy())) == [1]

    data = next(iter(img_data.predict_dataloader()))
    imgs = data["input"]
    assert imgs.shape == (1, 3, 196, 196)


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_from_filepaths_list_image_paths(tmpdir):
    tmpdir = Path(tmpdir)

    (tmpdir / "e").mkdir()
    _rand_image().save(tmpdir / "e_1.png")

    train_images = [
        str(tmpdir / "e_1.png"),
        str(tmpdir / "e_1.png"),
        str(tmpdir / "e_1.png"),
    ]

    img_data = ImageClassificationData.from_files(
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
    data = next(iter(img_data.train_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (2, 3, 196, 196)
    assert labels.shape == (2,)
    assert labels.numpy()[0] in [0, 3, 6]  # data comes shuffled here
    assert labels.numpy()[1] in [0, 3, 6]  # data comes shuffled here

    # check validation data
    data = next(iter(img_data.val_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (2, 3, 196, 196)
    assert labels.shape == (2,)
    assert list(labels.numpy()) == [1, 4]

    # check test data
    data = next(iter(img_data.test_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (2, 3, 196, 196)
    assert labels.shape == (2,)
    assert list(labels.numpy()) == [2, 5]


@pytest.mark.skipif(not _IMAGE_AVAILABLE, reason="image libraries aren't installed.")
@pytest.mark.skipif(not _MATPLOTLIB_AVAILABLE, reason="matplotlib isn't installed.")
def test_from_filepaths_visualise(tmpdir):
    tmpdir = Path(tmpdir)

    (tmpdir / "e").mkdir()
    _rand_image().save(tmpdir / "e_1.png")

    train_images = [
        str(tmpdir / "e_1.png"),
        str(tmpdir / "e_1.png"),
        str(tmpdir / "e_1.png"),
    ]

    dm = ImageClassificationData.from_files(
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


@pytest.mark.skipif(not _IMAGE_AVAILABLE, reason="image libraries aren't installed.")
@pytest.mark.skipif(not _MATPLOTLIB_AVAILABLE, reason="matplotlib isn't installed.")
def test_from_filepaths_visualise_subplots_exceding_max_cols(tmpdir):
    tmpdir = Path(tmpdir)

    (tmpdir / "e").mkdir()
    _rand_image().save(tmpdir / "e_1.png")

    train_images = [
        str(tmpdir / "e_1.png"),
    ] * 8

    dm = ImageClassificationData.from_files(
        train_files=train_images,
        train_targets=[0, 3, 6, 9, 8, 9, 1, 2],
        val_files=train_images,
        val_targets=[1, 4, 7, 8, 9, 8, 7, 1],
        test_files=train_images,
        test_targets=[2, 5, 8, 9, 7, 1, 2, 3],
        batch_size=8,
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


@pytest.mark.skipif(not _IMAGE_AVAILABLE, reason="image libraries aren't installed.")
@pytest.mark.skipif(not _MATPLOTLIB_AVAILABLE, reason="matplotlib isn't installed.")
def test_from_filepaths_visualise_subplots_single_image(tmpdir):
    tmpdir = Path(tmpdir)

    (tmpdir / "e").mkdir()
    _rand_image().save(tmpdir / "e_1.png")

    train_images = [
        str(tmpdir / "e_1.png"),
    ]

    dm = ImageClassificationData.from_files(
        train_files=train_images,
        train_targets=[0],
        val_files=train_images,
        val_targets=[1],
        test_files=train_images,
        test_targets=[2],
        batch_size=1,
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


@pytest.mark.skipif(not _IMAGE_AVAILABLE, reason="image libraries aren't installed.")
@pytest.mark.skipif(not _MATPLOTLIB_AVAILABLE, reason="matplotlib isn't installed.")
def test_from_filepaths_visualise_multilabel(tmpdir):
    tmpdir = Path(tmpdir)

    (tmpdir / "a").mkdir()
    (tmpdir / "b").mkdir()

    image_a = str(tmpdir / "a" / "a_1.png")
    image_b = str(tmpdir / "b" / "b_1.png")

    _rand_image().save(image_a)
    _rand_image().save(image_b)

    dm = ImageClassificationData.from_files(
        train_files=[image_a, image_b],
        train_targets=[[0, 1, 0], [0, 1, 1]],
        val_files=[image_b, image_a],
        val_targets=[[1, 1, 0], [0, 0, 1]],
        test_files=[image_b, image_b],
        test_targets=[[0, 0, 1], [1, 1, 0]],
        batch_size=2,
        transform_kwargs={"image_size": (64, 64)},
    )
    # disable visualisation for testing
    assert dm.data_fetcher.block_viz_window is True
    dm.set_block_viz_window(False)
    assert dm.data_fetcher.block_viz_window is False

    # call show functions
    dm.show_train_batch()
    dm.show_train_batch("per_sample_transform")
    dm.show_val_batch("per_batch_transform")


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_from_folders_only_train(tmpdir):
    train_dir = Path(tmpdir / "train")
    train_dir.mkdir()

    (train_dir / "a").mkdir()
    _rand_image().save(train_dir / "a" / "1.png")
    _rand_image().save(train_dir / "a" / "2.png")

    (train_dir / "b").mkdir()
    _rand_image().save(train_dir / "b" / "1.png")
    _rand_image().save(train_dir / "b" / "2.png")

    img_data = ImageClassificationData.from_folders(train_dir, batch_size=1)

    data = img_data.train_dataset[0]
    imgs, labels = data["input"], data["target"]
    assert isinstance(imgs, Image.Image)
    assert labels == 0


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_from_folders_train_val(tmpdir):
    train_dir = Path(tmpdir / "train")
    train_dir.mkdir()

    (train_dir / "a").mkdir()
    _rand_image().save(train_dir / "a" / "1.png")
    _rand_image().save(train_dir / "a" / "2.png")

    (train_dir / "b").mkdir()
    _rand_image().save(train_dir / "b" / "1.png")
    _rand_image().save(train_dir / "b" / "2.png")
    img_data = ImageClassificationData.from_folders(
        train_folder=train_dir,
        val_folder=train_dir,
        test_folder=train_dir,
        batch_size=2,
        num_workers=0,
    )

    data = next(iter(img_data.train_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (2, 3, 196, 196)
    assert labels.shape == (2,)

    data = next(iter(img_data.val_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (2, 3, 196, 196)
    assert labels.shape == (2,)
    assert list(labels.numpy()) == [0, 0]

    data = next(iter(img_data.test_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (2, 3, 196, 196)
    assert labels.shape == (2,)
    assert list(labels.numpy()) == [0, 0]


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_from_filepaths_multilabel(tmpdir):
    tmpdir = Path(tmpdir)

    (tmpdir / "a").mkdir()
    _rand_image().save(tmpdir / "a1.png")
    _rand_image().save(tmpdir / "a2.png")

    train_images = [str(tmpdir / "a1.png"), str(tmpdir / "a2.png")]
    train_labels = [[1, 0, 1, 0], [0, 0, 1, 1]]
    valid_labels = [[1, 1, 1, 0], [1, 0, 0, 1]]
    test_labels = [[1, 0, 1, 0], [1, 1, 0, 1]]

    dm = ImageClassificationData.from_files(
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
    assert imgs.shape == (2, 3, 196, 196)
    assert labels.shape == (2, 4)

    data = next(iter(dm.val_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (2, 3, 196, 196)
    assert labels.shape == (2, 4)
    torch.testing.assert_allclose(labels, torch.tensor(valid_labels))

    data = next(iter(dm.test_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (2, 3, 196, 196)
    assert labels.shape == (2, 4)
    torch.testing.assert_allclose(labels, torch.tensor(test_labels))


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
@pytest.mark.parametrize(
    "data,from_function",
    [
        (torch.rand(3, 3, 196, 196), ImageClassificationData.from_tensors),
        (np.random.rand(3, 3, 196, 196), ImageClassificationData.from_numpy),
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
    assert imgs.shape == (2, 3, 196, 196)
    assert labels.shape == (2,)
    assert labels.numpy()[0] in [0, 3, 6]  # data comes shuffled here
    assert labels.numpy()[1] in [0, 3, 6]  # data comes shuffled here

    # check validation data
    data = next(iter(img_data.val_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (2, 3, 196, 196)
    assert labels.shape == (2,)
    assert list(labels.numpy()) == [1, 4]

    # check test data
    data = next(iter(img_data.test_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (2, 3, 196, 196)
    assert labels.shape == (2,)
    assert list(labels.numpy()) == [2, 5]


@pytest.mark.skipif(not _IMAGE_AVAILABLE, reason="image libraries aren't installed.")
@pytest.mark.skipif(not _FIFTYONE_AVAILABLE, reason="fiftyone isn't installed.")
def test_from_fiftyone(tmpdir):
    tmpdir = Path(tmpdir)

    (tmpdir / "a").mkdir()
    (tmpdir / "b").mkdir()
    _rand_image().save(tmpdir / "a_1.png")
    _rand_image().save(tmpdir / "b_1.png")

    train_images = [
        str(tmpdir / "a_1.png"),
        str(tmpdir / "b_1.png"),
    ]

    dataset = fo.Dataset.from_dir(str(tmpdir), dataset_type=fo.types.ImageDirectory)
    s1 = dataset[train_images[0]]
    s2 = dataset[train_images[1]]
    s1["test"] = fo.Classification(label="1")
    s2["test"] = fo.Classification(label="2")
    s1.save()
    s2.save()

    img_data = ImageClassificationData.from_fiftyone(
        train_dataset=dataset,
        test_dataset=dataset,
        val_dataset=dataset,
        label_field="test",
        batch_size=2,
        num_workers=0,
    )
    assert img_data.train_dataloader() is not None
    assert img_data.val_dataloader() is not None
    assert img_data.test_dataloader() is not None

    # check train data
    data = next(iter(img_data.train_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (2, 3, 196, 196)
    assert labels.shape == (2,)
    assert sorted(list(labels.numpy())) == [0, 1]

    # check val data
    data = next(iter(img_data.val_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (2, 3, 196, 196)
    assert labels.shape == (2,)
    assert sorted(list(labels.numpy())) == [0, 1]

    # check test data
    data = next(iter(img_data.test_dataloader()))
    imgs, labels = data["input"], data["target"]
    assert imgs.shape == (2, 3, 196, 196)
    assert labels.shape == (2,)
    assert sorted(list(labels.numpy())) == [0, 1]


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_from_datasets():
    img_data = ImageClassificationData.from_datasets(
        train_dataset=FakeData(size=3, num_classes=2),
        val_dataset=FakeData(size=3, num_classes=2),
        test_dataset=FakeData(size=3, num_classes=2),
        batch_size=2,
        num_workers=0,
    )

    # check training data
    data = next(iter(img_data.train_dataloader()))
    imgs, labels = data[DataKeys.INPUT], data[DataKeys.TARGET]
    assert imgs.shape == (2, 3, 196, 196)
    assert labels.shape == (2,)

    # check validation data
    data = next(iter(img_data.val_dataloader()))
    imgs, labels = data[DataKeys.INPUT], data[DataKeys.TARGET]
    assert imgs.shape == (2, 3, 196, 196)
    assert labels.shape == (2,)

    # check test data
    data = next(iter(img_data.test_dataloader()))
    imgs, labels = data[DataKeys.INPUT], data[DataKeys.TARGET]
    assert imgs.shape == (2, 3, 196, 196)
    assert labels.shape == (2,)


@pytest.fixture
def image_tmpdir(tmpdir):
    (tmpdir / "train").mkdir()
    Image.new("RGB", (128, 128)).save(str(tmpdir / "train" / "image_1.png"))
    Image.new("RGB", (128, 128)).save(str(tmpdir / "train" / "image_2.png"))
    return tmpdir / "train"


@pytest.fixture
def single_target_csv(image_tmpdir):
    with open(image_tmpdir / "metadata.csv", "w") as csvfile:
        fieldnames = ["image", "target"]
        writer = csv.DictWriter(csvfile, fieldnames)
        writer.writeheader()
        writer.writerow({"image": "image_1.png", "target": "Ants"})
        writer.writerow({"image": "image_2.png", "target": "Bees"})
    return str(image_tmpdir / "metadata.csv")


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_from_csv_single_target(single_target_csv):
    img_data = ImageClassificationData.from_csv(
        "image",
        "target",
        train_file=single_target_csv,
        batch_size=2,
        num_workers=0,
    )

    # check training data
    data = next(iter(img_data.train_dataloader()))
    imgs, labels = data[DataKeys.INPUT], data[DataKeys.TARGET]
    assert imgs.shape == (2, 3, 196, 196)
    assert labels.shape == (2,)


@pytest.fixture
def multi_target_csv(image_tmpdir):
    with open(image_tmpdir / "metadata.csv", "w") as csvfile:
        fieldnames = ["image", "target_1", "target_2"]
        writer = csv.DictWriter(csvfile, fieldnames)
        writer.writeheader()
        writer.writerow({"image": "image_1.png", "target_1": 1, "target_2": 0})
        writer.writerow({"image": "image_2.png", "target_1": 1, "target_2": 1})
    return str(image_tmpdir / "metadata.csv")


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_from_csv_multi_target(multi_target_csv):
    img_data = ImageClassificationData.from_csv(
        "image",
        ["target_1", "target_2"],
        train_file=multi_target_csv,
        batch_size=2,
        num_workers=0,
    )

    # check training data
    data = next(iter(img_data.train_dataloader()))
    imgs, labels = data[DataKeys.INPUT], data[DataKeys.TARGET]
    assert imgs.shape == (2, 3, 196, 196)
    assert labels.shape == (2, 2)


@pytest.fixture
def bad_csv_no_image(image_tmpdir):
    with open(image_tmpdir / "metadata.csv", "w") as csvfile:
        fieldnames = ["image", "target"]
        writer = csv.DictWriter(csvfile, fieldnames)
        writer.writeheader()
        writer.writerow({"image": "image_3", "target": "Ants"})
    return str(image_tmpdir / "metadata.csv")


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_from_bad_csv_no_image(bad_csv_no_image):
    bad_file = os.path.join(os.path.dirname(bad_csv_no_image), "image_3")
    with pytest.raises(ValueError, match=f"File ID `image_3` resolved to `{bad_file}`, which does not exist."):
        img_data = ImageClassificationData.from_csv(
            "image",
            ["target"],
            train_file=bad_csv_no_image,
            batch_size=1,
            num_workers=0,
        )
        _ = next(iter(img_data.train_dataloader()))


@pytest.mark.skipif(not _IMAGE_AVAILABLE, reason="image libraries aren't installed.")
def test_mixup(single_target_csv):
    @dataclass
    class MyTransform(ImageClassificationInputTransform):

        alpha: float = 1.0

        def mixup(self, batch):
            images = batch["input"]
            targets = batch["target"].float().unsqueeze(1)

            lam = np.random.beta(self.alpha, self.alpha)
            perm = torch.randperm(images.size(0))

            batch["input"] = images * lam + images[perm] * (1 - lam)
            batch["target"] = targets * lam + targets[perm] * (1 - lam)
            for e in batch["metadata"]:
                e.update({"lam": lam})
            return batch

        def per_batch_transform(self):
            return self.mixup

    img_data = ImageClassificationData.from_csv(
        "image",
        "target",
        train_file=single_target_csv,
        batch_size=2,
        num_workers=0,
        transform=MyTransform,
    )

    batch = next(iter(img_data.train_dataloader()))
    assert "lam" in batch["metadata"][0]
