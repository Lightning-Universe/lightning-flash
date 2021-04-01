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

import kornia as K
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch import nn

from flash.vision import ImageClassificationData


def _rand_image():
    return Image.fromarray(np.random.randint(0, 255, (196, 196, 3), dtype="uint8"))


class ImageClassificationDataViz(ImageClassificationData):

    def show_train_batch(self):
        self.viz.enabled = True
        _ = next(iter(self.train_dataloader()))
        self.viz.enabled = False


def test_base_viz(tmpdir):
    tmpdir = Path(tmpdir)

    (tmpdir / "a").mkdir()
    (tmpdir / "b").mkdir()
    _rand_image().save(tmpdir / "a" / "a_1.png")
    _rand_image().save(tmpdir / "a" / "a_2.png")

    _rand_image().save(tmpdir / "b" / "a_1.png")
    _rand_image().save(tmpdir / "b" / "a_2.png")

    img_data = ImageClassificationDataViz.from_filepaths(
        train_filepaths=[tmpdir / "a", tmpdir / "b"],
        train_labels=[0, 1],
        batch_size=1,
        num_workers=0,
    )

    img_data.show_train_batch()
    assert img_data.viz.batches["train"]["load_sample"] is not None
    assert img_data.viz.batches["train"]["to_tensor_transform"] is not None
    assert img_data.viz.batches["train"]["collate"] is not None
    assert img_data.viz.batches["train"]["per_batch_transform"] is not None


def test_base_viz_kornia(tmpdir):
    tmpdir = Path(tmpdir)

    (tmpdir / "a").mkdir()
    (tmpdir / "b").mkdir()
    _rand_image().save(tmpdir / "a" / "a_1.png")
    _rand_image().save(tmpdir / "a" / "a_2.png")

    _rand_image().save(tmpdir / "b" / "a_1.png")
    _rand_image().save(tmpdir / "b" / "a_2.png")

    # Define the augmentations pipeline
    train_transforms = {
        # can we just call this `preprocess` ?
        # this is needed all the time in train, valid, etc
        "pre_tensor_transform": T.Compose([T.RandomResizedCrop(224), T.ToTensor()]),
        "post_tensor_transform": nn.Sequential(
            # Kornia RandomResizeCrop has a bug - I'll debug with Jian.
            # K.augmentation.RandomResizedCrop((224, 224), align_corners=True),
            K.augmentation.Normalize(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])),
        ),
        "per_batch_transform": nn.Sequential(
            K.augmentation.RandomAffine(360., p=0.5), K.augmentation.ColorJitter(0.2, 0.3, 0.2, 0.3, p=0.5)
        )
    }
    img_data = ImageClassificationDataViz.from_filepaths(
        train_filepaths=[tmpdir / "a", tmpdir / "b"],
        train_labels=[0, 1],
        batch_size=1,
        num_workers=0,
        train_transform=train_transforms,
        valt_transform=train_transforms,
    )

    img_data.show_train_batch()
    assert img_data.viz.batches["train"]["load_sample"] is not None
    assert img_data.viz.batches["train"]["to_tensor_transform"] is not None
    assert img_data.viz.batches["train"]["collate"] is not None
    assert img_data.viz.batches["train"]["per_batch_transform"] is not None
