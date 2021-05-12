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
from flash import Trainer
from flash.data.utils import download_data
from flash.vision import ObjectDetector

# 1. Download the data
# Dataset Credit: https://www.kaggle.com/ultralytics/coco128
download_data("https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.3.0/coco128.zip", "data/")

# 2. Load the model from a checkpoint
model = ObjectDetector.load_from_checkpoint("https://flash-weights.s3.amazonaws.com/object_detection_model.pt")

# 3. Detect the object on the images
predictions = model.predict([
    "data/coco128/images/train2017/000000000025.jpg",
    "data/coco128/images/train2017/000000000520.jpg",
    "data/coco128/images/train2017/000000000532.jpg",
])
print(predictions)
