from typing import Tuple

import torch.nn as nn
import torchvision
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def torchvision_backbone_and_num_features(model_name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:

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

    else:
        raise ValueError(f"{model_name} is not supported yet.")
