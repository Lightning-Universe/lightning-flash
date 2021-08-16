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
from flash.core.registry import Provider

_ICEVISION = Provider("airctic/IceVision", "https://github.com/airctic/icevision")
_TORCHVISION = Provider("PyTorch/torchvision", "https://github.com/pytorch/vision")
_ULTRALYTICS = Provider("Ultralytics/YOLOV5", "https://github.com/ultralytics/yolov5")
_MMDET = Provider("OpenMMLab/MMDetection", "https://github.com/open-mmlab/mmdetection")
_EFFDET = Provider("rwightman/efficientdet-pytorch", "https://github.com/rwightman/efficientdet-pytorch")
