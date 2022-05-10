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
import collections
import json
import os
from typing import Any

import numpy as np
import pytest
import torch

from flash import Trainer
from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _ICEVISION_AVAILABLE, _IMAGE_AVAILABLE
from flash.image import KeypointDetectionData, KeypointDetector
from tests.helpers.task_tester import TaskTester

if _IMAGE_AVAILABLE:
    from PIL import Image

COCODataConfig = collections.namedtuple("COCODataConfig", "train_folder train_ann_file predict_folder")


@pytest.fixture
def coco_keypoints(tmpdir):
    rand_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))
    os.makedirs(tmpdir / "train_folder", exist_ok=True)
    os.makedirs(tmpdir / "predict_folder", exist_ok=True)

    train_folder = tmpdir / "train_folder"
    train_ann_file = tmpdir / "train_annotations.json"
    predict_folder = tmpdir / "predict_folder"

    _ = [rand_image.save(str(train_folder / f"image_{i}.png")) for i in range(1, 4)]
    _ = [rand_image.save(str(predict_folder / f"predict_image_{i}.png")) for i in range(1, 4)]
    annotations = {
        "annotations": [
            {
                "area": 50,
                "bbox": [10, 20, 5, 10],
                "num_keypoints": 2,
                "keypoints": [10, 15, 2, 20, 30, 2],
                "category_id": 1,
                "id": 1,
                "image_id": 1,
                "iscrowd": 0,
            },
            {
                "area": 100,
                "bbox": [20, 30, 10, 10],
                "num_keypoints": 2,
                "keypoints": [20, 30, 2, 30, 40, 2],
                "category_id": 2,
                "id": 2,
                "image_id": 2,
                "iscrowd": 0,
            },
            {
                "area": 125,
                "bbox": [10, 20, 5, 25],
                "num_keypoints": 2,
                "keypoints": [10, 15, 2, 20, 45, 2],
                "category_id": 1,
                "id": 3,
                "image_id": 3,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 1, "name": "cat", "supercategory": "cat", "keypoints": ["left ear", "right ear"]},
            {"id": 2, "name": "dog", "supercategory": "dog", "keypoints": ["left ear", "right ear"]},
        ],
        "images": [
            {"file_name": "image_1.png", "height": 64, "width": 64, "id": 1},
            {"file_name": "image_2.png", "height": 64, "width": 64, "id": 2},
            {"file_name": "image_3.png", "height": 64, "width": 64, "id": 3},
        ],
    }
    with open(train_ann_file, "w") as annotation_file:
        json.dump(annotations, annotation_file)

    return COCODataConfig(train_folder, train_ann_file, predict_folder)


class TestKeypointDetector(TaskTester):

    task = KeypointDetector
    task_args = (2,)
    task_kwargs = {"num_classes": 2}
    cli_command = "keypoint_detection"
    is_testing = _IMAGE_AVAILABLE and _ICEVISION_AVAILABLE
    is_available = _IMAGE_AVAILABLE and _ICEVISION_AVAILABLE

    # TODO: Resolve JIT support
    traceable = False
    scriptable = False

    @property
    def example_forward_input(self):
        return torch.rand(1, 3, 32, 32)

    def check_forward_output(self, output: Any):
        assert {"keypoints", "labels", "scores"} <= output[0].keys()

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
                "keypoints": [
                    [{"x": 10, "y": 10, "visible": 1}],
                    [{"x": 10, "y": 10, "visible": 1}],
                ],
            },
        }

    @property
    def example_val_sample(self):
        return self.example_train_sample

    @property
    def example_test_sample(self):
        return self.example_train_sample


@pytest.mark.skipif(not _IMAGE_AVAILABLE, reason="image libraries aren't installed.")
@pytest.mark.skipif(not _ICEVISION_AVAILABLE, reason="IceVision is not installed for testing")
@pytest.mark.parametrize("backbone, head", [("resnet18_fpn", "keypoint_rcnn")])
def test_model(coco_keypoints, backbone, head):
    datamodule = KeypointDetectionData.from_coco(
        train_folder=coco_keypoints.train_folder,
        train_ann_file=coco_keypoints.train_ann_file,
        predict_folder=coco_keypoints.predict_folder,
        transform_kwargs=dict(image_size=(128, 128)),
        batch_size=2,
    )

    assert datamodule.num_classes == 3
    assert datamodule.labels == ["background", "cat", "dog"]

    model = KeypointDetector(2, num_classes=datamodule.num_classes, backbone=backbone, head=head)
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model, datamodule=datamodule)
    trainer.predict(model, datamodule=datamodule)
