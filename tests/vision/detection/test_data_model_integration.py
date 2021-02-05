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
import flash
from flash.vision import ImageDetector
from flash.vision.detection.data import ImageDetectionData
from tests.vision.detection.test_data import _create_synth_coco_dataset


def test_detection(tmpdir):

    train_folder, coco_ann_path = _create_synth_coco_dataset(tmpdir)

    data = ImageDetectionData.from_coco(train_folder=train_folder, train_ann_file=coco_ann_path, batch_size=1)
    model = ImageDetector(num_classes=data.num_classes)

    trainer = flash.Trainer(fast_dev_run=True)

    trainer.finetune(model, data)
