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
import random
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from flash.core.data.io.input import DataKeys
from flash.core.integrations.icevision.transforms import IceVisionInputTransform
from flash.core.trainer import Trainer
from flash.core.utilities.imports import (
    _EFFDET_AVAILABLE,
    _ICEVISION_AVAILABLE,
    _TOPIC_IMAGE_AVAILABLE,
    _TOPIC_SERVE_AVAILABLE,
)
from flash.image import ObjectDetector
from torch.utils.data import Dataset

from tests.helpers.task_tester import TaskTester


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
        xs = torch.randint(w - 1, (2,))
        ys = torch.randint(h - 1, (2,))
        return {"xmin": min(xs), "ymin": min(ys), "width": max(xs) - min(xs) + 1, "height": max(ys) - min(ys) + 1}

    def __getitem__(self, idx):
        sample = {}

        img = np.random.rand(*self.img_shape).astype(np.float32)

        sample[DataKeys.INPUT] = img

        sample[DataKeys.TARGET] = {
            "bboxes": [],
            "labels": [],
        }

        for i in range(self.num_boxes):
            sample[DataKeys.TARGET]["bboxes"].append(self._random_bbox())
            sample[DataKeys.TARGET]["labels"].append(random.randint(0, self.num_classes - 1))

        return sample


@pytest.mark.skipif(not _EFFDET_AVAILABLE, reason="effdet is not installed for testing")
class TestObjectDetector(TaskTester):
    task = ObjectDetector
    task_kwargs = {"num_classes": 2}
    cli_command = "object_detection"
    is_testing = _TOPIC_IMAGE_AVAILABLE
    is_available = _TOPIC_IMAGE_AVAILABLE and _ICEVISION_AVAILABLE

    # TODO: Resolve JIT support
    traceable = False
    scriptable = False

    @property
    def example_forward_input(self):
        return torch.rand(1, 3, 32, 32)

    def check_forward_output(self, output: Any):
        assert {"bboxes", "labels", "scores"} <= output[0].keys()

    @property
    def example_train_sample(self):
        return {
            DataKeys.INPUT: torch.rand(3, 224, 224),
            DataKeys.TARGET: {
                "bboxes": [
                    {"xmin": 10, "ymin": 10, "width": 20, "height": 20},
                    {"xmin": 30, "ymin": 30, "width": 40, "height": 40},
                ],
                "labels": [0, 1],
            },
        }

    @property
    def example_val_sample(self):
        return self.example_train_sample

    @property
    def example_test_sample(self):
        return self.example_train_sample


@pytest.mark.parametrize("head", ["retinanet"])
@pytest.mark.skipif(not _TOPIC_IMAGE_AVAILABLE, reason="image libraries aren't installed.")
def test_predict(tmpdir, head):
    model = ObjectDetector(num_classes=2, head=head, pretrained=False)
    ds = DummyDetectionDataset((128, 128, 3), 1, 2, 10)

    input_transform = IceVisionInputTransform()

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    dl = model.process_train_dataset(
        ds,
        2,
        num_workers=0,
        pin_memory=False,
        input_transform=input_transform,
    )
    trainer.fit(model, dl)

    dl = model.process_predict_dataset(ds, 2, input_transform=input_transform)
    predictions = trainer.predict(model, dl, output="preds")
    assert len(predictions[0][0]["bboxes"]) > 0
    model.predict_kwargs = {"detection_threshold": 2}
    predictions = trainer.predict(model, dl, output="preds")
    assert len(predictions[0][0]["bboxes"]) == 0


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="serve libraries aren't installed.")
@patch("flash._IS_TESTING", True)
def test_serve():
    model = ObjectDetector(2)
    model.eval()
    model.serve()
