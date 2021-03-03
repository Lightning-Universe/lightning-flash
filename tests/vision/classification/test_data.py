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

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from flash.data.data_utils import labels_from_categorical_csv
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


def test_categorical_csv_labels(tmpdir):
    train_dir = Path(tmpdir / "some_dataset")
    train_dir.mkdir()

    (train_dir / "train").mkdir()
    _rand_image().save(train_dir / "train" / "train_1.png")
    _rand_image().save(train_dir / "train" / "train_2.png")

    (train_dir / "valid").mkdir()
    _rand_image().save(train_dir / "valid" / "valid_1.png")
    _rand_image().save(train_dir / "valid" / "valid_2.png")

    (train_dir / "test").mkdir()
    _rand_image().save(train_dir / "test" / "test_1.png")
    _rand_image().save(train_dir / "test" / "test_2.png")

    train_csv = os.path.join(tmpdir, 'some_dataset', 'train.csv')
    text_file = open(train_csv, 'w')
    text_file.write(
        'my_id,label_a,label_b,label_c\n"train_1.png", 0, 1, 0\n"train_2.png", 0, 0, 1\n"train_2.png", 1, 0, 0\n'
    )
    text_file.close()

    valid_csv = os.path.join(tmpdir, 'some_dataset', 'valid.csv')
    text_file = open(valid_csv, 'w')
    text_file.write(
        'my_id,label_a,label_b,label_c\n"valid_1.png", 0, 1, 0\n"valid_2.png", 0, 0, 1\n"valid_3.png", 1, 0, 0\n'
    )
    text_file.close()

    test_csv = os.path.join(tmpdir, 'some_dataset', 'test.csv')
    text_file = open(test_csv, 'w')
    text_file.write(
        'my_id,label_a,label_b,label_c\n"test_1.png", 0, 1, 0\n"test_2.png", 0, 0, 1\n"test_3.png", 1, 0, 0\n'
    )
    text_file.close()

    def index_col_collate_fn(x):
        return os.path.splitext(x)[0]

    train_labels = labels_from_categorical_csv(
        train_csv, 'my_id', feature_cols=['label_a', 'label_b', 'label_c'], index_col_collate_fn=index_col_collate_fn
    )
    valid_labels = labels_from_categorical_csv(
        valid_csv, 'my_id', feature_cols=['label_a', 'label_b', 'label_c'], index_col_collate_fn=index_col_collate_fn
    )
    test_labels = labels_from_categorical_csv(
        test_csv, 'my_id', feature_cols=['label_a', 'label_b', 'label_c'], index_col_collate_fn=index_col_collate_fn
    )

    data = ImageClassificationData.from_filepaths(
        batch_size=2,
        train_filepaths=os.path.join(tmpdir, 'some_dataset', 'train'),
        train_labels=train_labels,
        valid_filepaths=os.path.join(tmpdir, 'some_dataset', 'valid'),
        valid_labels=valid_labels,
        test_filepaths=os.path.join(tmpdir, 'some_dataset', 'test'),
        test_labels=test_labels,
    )

    for (x, y) in data.train_dataloader():
        assert len(x) == 2

    for (x, y) in data.val_dataloader():
        assert len(x) == 2

    for (x, y) in data.test_dataloader():
        assert len(x) == 2

    data = ImageClassificationData.from_filepaths(
        batch_size=2,
        train_filepaths=os.path.join(tmpdir, 'some_dataset', 'train'),
        train_labels=train_labels,
        valid_split=0.5
    )

    for (x, y) in data.val_dataloader():
        assert len(x) == 1


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
