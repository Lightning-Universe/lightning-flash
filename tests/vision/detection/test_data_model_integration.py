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
from PIL import Image
from pytorch_lightning.utilities import _module_available

import flash
from flash.vision import ObjectDetector
from flash.vision.detection import ObjectDetectionData
from tests.vision.detection.test_data import _create_synth_coco_dataset

_COCO_AVAILABLE = _module_available("pycocotools")


@pytest.mark.skipif(not _COCO_AVAILABLE, reason="pycocotools is not installed for testing")
@pytest.mark.parametrize(["model", "backbone"], [("fasterrcnn", None), ("retinanet", "resnet34"),
                                                 ("fasterrcnn", "mobilenet_v2"), ("retinanet", "simclr-imagenet")])
def test_detection(tmpdir, model, backbone):

    train_folder, coco_ann_path = _create_synth_coco_dataset(tmpdir)

    data = ObjectDetectionData.from_coco(train_folder=train_folder, train_ann_file=coco_ann_path, batch_size=1)
    model = ObjectDetector(model=model, backbone=backbone, num_classes=data.num_classes)

    trainer = flash.Trainer(fast_dev_run=True)

    trainer.finetune(model, data)

    test_image_one = os.fspath(tmpdir / "test_one.png")
    test_image_two = os.fspath(tmpdir / "test_two.png")

    Image.new('RGB', (1920, 1080)).save(test_image_one)
    Image.new('RGB', (1920, 1080)).save(test_image_two)

    test_images = [test_image_one, test_image_two]

    model.predict(test_images)
