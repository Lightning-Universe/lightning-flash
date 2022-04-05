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

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from flash.core.data.callback import BaseDataFetcher
from flash.core.data.io.input import DataKeys
from flash.core.data.io.input_transform import create_worker_input_transform_processor
from flash.core.integrations.icevision.transforms import IceVisionInputTransform
from flash.core.trainer import Trainer
from flash.core.utilities.imports import _ICEVISION_AVAILABLE, _IMAGE_AVAILABLE
from flash.core.utilities.stages import RunningStage
from flash.image import ObjectDetector
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


class TestObjectDetector(TaskTester):

    task = ObjectDetector
    task_kwargs = {"num_classes": 2}
    cli_command = "object_detection"
    is_testing = _IMAGE_AVAILABLE and _ICEVISION_AVAILABLE
    is_available = _IMAGE_AVAILABLE and _ICEVISION_AVAILABLE

    # TODO: Resolve JIT support
    traceable = False
    scriptable = False

    @property
    def example_forward_input(self):
        return [torch.rand(3, 32, 32)]

    def check_forward_output(self, output: Any):
        assert {"boxes", "labels", "scores"} <= output[0].keys()


@pytest.mark.skipif(not _IMAGE_AVAILABLE, reason="image libraries aren't installed.")
@pytest.mark.skipif(not _ICEVISION_AVAILABLE, reason="IceVision is not installed for testing")
def test_init():
    model = ObjectDetector(num_classes=2)
    model.eval()

    batch_size = 2
    ds = DummyDetectionDataset((128, 128, 3), 1, 2, 10)
    input_transform = IceVisionInputTransform()
    predict_collate_fn = create_worker_input_transform_processor(
        RunningStage.PREDICTING, input_transform, [BaseDataFetcher()]
    )
    dl = model.process_predict_dataset(
        dataset=ds,
        input_transform=input_transform,
        batch_size=batch_size,
        collate_fn=predict_collate_fn,
    )
    data = next(iter(dl))

    out = model.forward(data[DataKeys.INPUT])

    assert len(out) == batch_size
    assert all(isinstance(res, dict) for res in out)
    assert all("bboxes" in res for res in out)
    assert all("labels" in res for res in out)
    assert all("scores" in res for res in out)


@pytest.mark.parametrize("head", ["faster_rcnn", "retinanet"])
@pytest.mark.skipif(not _IMAGE_AVAILABLE, reason="image libraries aren't installed.")
@pytest.mark.skipif(not _ICEVISION_AVAILABLE, reason="IceVision is not installed for testing")
def test_training(tmpdir, head):
    model = ObjectDetector(num_classes=2, head=head, pretrained=False)
    ds = DummyDetectionDataset((128, 128, 3), 1, 2, 10)

    input_transform = IceVisionInputTransform()
    data_fetcher = BaseDataFetcher()
    train_collate_fn = create_worker_input_transform_processor(RunningStage.TRAINING, input_transform, [data_fetcher])

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    dl = model.process_train_dataset(
        dataset=ds,
        trainer=trainer,
        input_transform=input_transform,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        collate_fn=train_collate_fn,
    )
    trainer.fit(model, dl)


@pytest.mark.parametrize("head", ["retinanet"])
@pytest.mark.skipif(not _IMAGE_AVAILABLE, reason="image libraries aren't installed.")
@pytest.mark.skipif(not _ICEVISION_AVAILABLE, reason="IceVision is not installed for testing")
def test_predict(tmpdir, head):
    model = ObjectDetector(num_classes=2, head=head, pretrained=False)
    ds = DummyDetectionDataset((128, 128, 3), 1, 2, 10)

    input_transform = IceVisionInputTransform()
    data_fetcher = BaseDataFetcher()
    train_collate_fn = create_worker_input_transform_processor(RunningStage.TRAINING, input_transform, [data_fetcher])
    predict_collate_fn = create_worker_input_transform_processor(
        RunningStage.PREDICTING, input_transform, [data_fetcher]
    )

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    dl = model.process_train_dataset(
        dataset=ds,
        trainer=trainer,
        input_transform=input_transform,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        collate_fn=train_collate_fn,
    )
    trainer.fit(model, dl)

    dl = model.process_predict_dataset(
        dataset=ds,
        input_transform=input_transform,
        batch_size=2,
        collate_fn=predict_collate_fn,
    )
    predictions = trainer.predict(model, dl, output="preds")
    assert len(predictions[0][0]["bboxes"]) > 0
    model.predict_kwargs = {"detection_threshold": 2}
    predictions = trainer.predict(model, dl, output="preds")
    assert len(predictions[0][0]["bboxes"]) == 0
