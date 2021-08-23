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
from dataclasses import dataclass

PROVIDERS = []  #: testing


@dataclass
class Provider:

    name: str
    url: str

    def __post_init__(self):
        PROVIDERS.append(self)

    def __str__(self):
        return f"{self.name} ({self.url})"


_TIMM = Provider("rwightman/pytorch-image-models", "https://github.com/rwightman/pytorch-image-models")
_DINO = Provider("Facebook Research/dino", "https://github.com/facebookresearch/dino")
_ICEVISION = Provider("airctic/IceVision", "https://github.com/airctic/icevision")
_TORCHVISION = Provider("PyTorch/torchvision", "https://github.com/pytorch/vision")
_ULTRALYTICS = Provider("Ultralytics/YOLOV5", "https://github.com/ultralytics/yolov5")
_MMDET = Provider("OpenMMLab/MMDetection", "https://github.com/open-mmlab/mmdetection")
_EFFDET = Provider("rwightman/efficientdet-pytorch", "https://github.com/rwightman/efficientdet-pytorch")
_SEGMENTATION_MODELS = Provider(
    "qubvel/segmentation_models.pytorch", "https://github.com/qubvel/segmentation_models.pytorch"
)
_PYSTICHE = Provider("pystiche/pystiche", "https://github.com/pystiche/pystiche")
_HUGGINGFACE = Provider("Hugging Face/transformers", "https://github.com/huggingface/transformers")
_FAIRSEQ = Provider("PyTorch/fairseq", "https://github.com/pytorch/fairseq")
_OPEN3D_ML = Provider("Intelligent Systems Lab Org/Open3D-ML", "https://github.com/isl-org/Open3D-ML")
_PYTORCHVIDEO = Provider("Facebook Research/PyTorchVideo", "https://github.com/facebookresearch/pytorchvideo")
