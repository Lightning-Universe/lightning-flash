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

import pytest
import torch

import flash
from flash.core.utilities.imports import _COCO_AVAILABLE, _FIFTYONE_AVAILABLE, _IMAGE_EXTRAS_TESTING, _PIL_AVAILABLE
from flash.image import ObjectDetector
from flash.image.detection import ObjectDetectionData

if _PIL_AVAILABLE:
    from PIL import Image
else:
    Image = None

if _COCO_AVAILABLE:
    from tests.image.detection.test_data import _create_synth_coco_dataset

if _FIFTYONE_AVAILABLE:
    from tests.image.detection.test_data import _create_synth_fiftyone_dataset


@pytest.mark.skipif(not _IMAGE_EXTRAS_TESTING, reason="image libraries aren't installed.")
@pytest.mark.parametrize(["head", "backbone"], [("retinanet", "resnet18_fpn"), ("faster_rcnn", "resnet18_fpn")])
def test_detection(tmpdir, head, backbone):
    train_folder, coco_ann_path = _create_synth_coco_dataset(tmpdir)

    datamodule = ObjectDetectionData.from_coco(train_folder=train_folder, train_ann_file=coco_ann_path, batch_size=1)
    model = ObjectDetector(head=head, backbone=backbone, num_classes=datamodule.num_classes)

    trainer = flash.Trainer(fast_dev_run=True, gpus=torch.cuda.device_count())

    trainer.finetune(model, datamodule=datamodule, strategy="freeze")

    test_image_one = os.fspath(tmpdir / "test_one.png")
    test_image_two = os.fspath(tmpdir / "test_two.png")

    Image.new("RGB", (512, 512)).save(test_image_one)
    Image.new("RGB", (512, 512)).save(test_image_two)

    datamodule = ObjectDetectionData.from_files(predict_files=[str(test_image_one), str(test_image_two)], batch_size=1)
    trainer.predict(model, datamodule=datamodule)


@pytest.mark.skipif(not _IMAGE_EXTRAS_TESTING, reason="image libraries aren't installed.")
@pytest.mark.skipif(not _FIFTYONE_AVAILABLE, reason="fiftyone is not installed for testing")
@pytest.mark.parametrize(["head", "backbone"], [("retinanet", "resnet18_fpn")])
def test_detection_fiftyone(tmpdir, head, backbone):
    train_dataset = _create_synth_fiftyone_dataset(tmpdir)

    datamodule = ObjectDetectionData.from_fiftyone(train_dataset=train_dataset, batch_size=1)
    model = ObjectDetector(head=head, backbone=backbone, num_classes=datamodule.num_classes)

    trainer = flash.Trainer(fast_dev_run=True, gpus=torch.cuda.device_count())

    trainer.finetune(model, datamodule, strategy="freeze")

    test_image_one = os.fspath(tmpdir / "test_one.png")
    test_image_two = os.fspath(tmpdir / "test_two.png")

    Image.new("RGB", (512, 512)).save(test_image_one)
    Image.new("RGB", (512, 512)).save(test_image_two)

    datamodule = ObjectDetectionData.from_files(predict_files=[str(test_image_one), str(test_image_two)], batch_size=1)
    trainer.predict(model, datamodule=datamodule)
