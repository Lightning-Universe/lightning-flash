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
from flash.core.data import download_data


def coco128_data_download(path: str):
    # Dataset Credit: https://www.kaggle.com/ultralytics/coco128
    URL = "https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.3.0/coco128.zip"
    download_data(URL, path)
