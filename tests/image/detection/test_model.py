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
import random
import re

import numpy as np
import pytest
import torch
from pytorch_lightning import Trainer
from torch.utils.data import Dataset

from flash.core.utilities.imports import _ICEVISION_AVAILABLE, _IMAGE_AVAILABLE
from flash.image import ObjectDetector
from tests.helpers.utils import _IMAGE_TESTING

if _ICEVISION_AVAILABLE:
    from icevision.core import BBox, ClassMap, ObjectDetectionRecord
    from icevision.data import Prediction
    from icevision.utils import ImgSize


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
        record = ObjectDetectionRecord()

        img = np.random.rand(*self.img_shape).astype(np.float32)

        record.set_img(img)
        record.set_img_size(ImgSize(width=self.img_shape[0], height=self.img_shape[1]))
        record.detection.set_class_map(ClassMap([f"test_{i}" for i in range(self.num_classes)], background=None))

        for i in range(self.num_boxes):
            record.detection.add_bboxes([BBox.from_xyxy(*self._random_bbox())])
            record.detection.add_labels([f"test_{random.randint(0, self.num_classes - 1)}"])

        return record


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_init():
    model = ObjectDetector(num_classes=2)
    model.eval()

    batch_size = 2
    ds = DummyDetectionDataset((128, 128, 3), 1, 2, 10)
    dl = model.process_predict_dataset(ds, batch_size=batch_size)
    data = next(iter(dl))

    out = model(data)

    assert len(out) == batch_size
    assert all(isinstance(res, Prediction) for res in out)


@pytest.mark.parametrize("head", ["faster_rcnn", "retinanet"])
@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_training(tmpdir, head):
    model = ObjectDetector(num_classes=2, head=head, pretrained=False)
    ds = DummyDetectionDataset((128, 128, 3), 1, 2, 10)
    dl = model.process_train_dataset(ds, 2, 0, False)
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
