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
import torch.nn as nn

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _TORCHVISION_AVAILABLE

SEMANTIC_SEGMENTATION_BACKBONES = FlashRegistry("backbones")

if _TORCHVISION_AVAILABLE:
    import torchvision

    @SEMANTIC_SEGMENTATION_BACKBONES(name="torchvision/fcn_resnet50")
    def load_torchvision_fcn_resnet50(num_classes: int, pretrained: bool = True) -> nn.Module:
        model = torchvision.models.segmentation.fcn_resnet50(pretrained=pretrained)
        model.classifier[-1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        return model

    @SEMANTIC_SEGMENTATION_BACKBONES(name="torchvision/fcn_resnet101")
    def load_torchvision_fcn_resnet101(num_classes: int, pretrained: bool = True) -> nn.Module:
        model = torchvision.models.segmentation.fcn_resnet101(pretrained=pretrained)
        model.classifier[-1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        return model
