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
import re

import pytest
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset

from flash.core.data.data_source import DefaultDataKeys
from flash.core.utilities.imports import _IMAGE_AVAILABLE
from flash.image import ObjectDetector
from tests.helpers.utils import _IMAGE_TESTING


def collate_fn(samples):
    return {key: [sample[key] for sample in samples] for key in samples[0]}


class DummyDetectionDataset(Dataset):

    def __init__(self, img_shape, num_boxes, num_classes, length):
        super().__init__()
        self.img_shape = img_shape
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.length = length

    def __len__(self) -> int:
        return self.length

    def _random_bbox(self):
        c, h, w = self.img_shape
        xs = torch.randint(w - 1, (2, ))
        ys = torch.randint(h - 1, (2, ))
        return [min(xs), min(ys), max(xs) + 1, max(ys) + 1]

    def __getitem__(self, idx):
        img = torch.rand(self.img_shape)
        boxes = torch.tensor([self._random_bbox() for _ in range(self.num_boxes)])
        labels = torch.randint(self.num_classes, (self.num_boxes, ))
        return {DefaultDataKeys.INPUT: img, DefaultDataKeys.TARGET: {"boxes": boxes, "labels": labels}}


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_init():
    model = ObjectDetector(num_classes=2)
    model.eval()

    batch_size = 2
    ds = DummyDetectionDataset((3, 224, 224), 1, 2, 10)
    dl = DataLoader(ds, collate_fn=collate_fn, batch_size=batch_size)
    data = next(iter(dl))
    img = data[DefaultDataKeys.INPUT]

    out = model(img)

    assert len(out) == batch_size
    assert {"boxes", "labels", "scores"} <= out[0].keys()


@pytest.mark.parametrize("model", ["fasterrcnn", "retinanet"])
@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_training(tmpdir, model):
    model = ObjectDetector(num_classes=2, model=model, pretrained=False, pretrained_backbone=False)
    ds = DummyDetectionDataset((3, 224, 224), 1, 2, 10)
    dl = DataLoader(ds, collate_fn=collate_fn)
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model, dl)


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_jit(tmpdir):
    path = os.path.join(tmpdir, "test.pt")

    model = ObjectDetector(2)
    model.eval()

    model = torch.jit.script(model)  # torch.jit.trace doesn't work with torchvision RCNN

    torch.jit.save(model, path)
    model = torch.jit.load(path)

    out = model([torch.rand(3, 32, 32)])

    # torchvision RCNN always returns a (Losses, Detections) tuple in scripting
    out = out[1]

    assert {"boxes", "labels", "scores"} <= out[0].keys()


@pytest.mark.skipif(_IMAGE_AVAILABLE, reason="image libraries are installed.")
def test_load_from_checkpoint_dependency_error():
    with pytest.raises(ModuleNotFoundError, match=re.escape("'lightning-flash[image]'")):
        ObjectDetector.load_from_checkpoint("not_a_real_checkpoint.pt")
