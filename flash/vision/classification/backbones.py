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
from typing import Tuple

import torch.nn as nn
import torchvision
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def torchvision_backbone_and_num_features(model_name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    >>> torchvision_backbone_and_num_features('mobilenet_v2')  # doctest: +ELLIPSIS
    ...
    >>> torchvision_backbone_and_num_features('resnet18')  # doctest: +ELLIPSIS
    ...
    >>> torchvision_backbone_and_num_features('densenet121')  # doctest: +ELLIPSIS
    ...
    """

    model = getattr(torchvision.models, model_name, None)
    if model is None:
        raise MisconfigurationException(f"{model_name} is not supported by torchvision")

    if model_name in ["mobilenet_v2", "vgg11", "vgg13", "vgg16", "vgg19"]:
        model = model(pretrained=pretrained)
        backbone = model.features
        num_features = model.classifier[-1].in_features
        return backbone, num_features

    elif model_name in [
        "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x8d"
    ]:
        model = model(pretrained=pretrained)
        # remove the last two layers & turn it into a Sequential model
        backbone = nn.Sequential(*list(model.children())[:-2])
        num_features = model.fc.in_features
        return backbone, num_features

    elif model_name in ["densenet121", "densenet169", "densenet161", "densenet161"]:
        model = model(pretrained=pretrained)
        backbone = nn.Sequential(*model.features, nn.ReLU(inplace=True))
        num_features = model.classifier.in_features
        return backbone, num_features

    raise ValueError(f"{model_name} is not supported yet.")
