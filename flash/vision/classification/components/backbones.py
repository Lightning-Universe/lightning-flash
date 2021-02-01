from typing import Tuple

import torch.nn as nn

from flash.vision.classification.components.torchvision_model_zoo import TORCHVISION_MODEL_ZOO


def torchvision_backbone_and_num_features(model_name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    Returns CNN backbone & it's final num of features from Torchvision supported models.
    Args:
        model_name: Name of the model. E.g. resnet18
        pretrained: Pretrained weights on the ImageNet dataset
    """
    model = TORCHVISION_MODEL_ZOO[model_name]
    model = model(pretrained=pretrained)
    if model_name in ["mobilenet_v2", "vgg11", "vgg13", "vgg16", "vgg19"]:
        backbone = model.features
        num_features = model.classifier[-1].in_features
        return backbone, num_features

    elif model_name in [
        "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x8d"
    ]:
        backbone = nn.Sequential(*list(model.children())[:-2])
        num_features = model.fc.in_features
        return backbone, num_features

    elif model_name in ["dense121", "densenet169", "densenet161", "densenet161"]:
        backbone = nn.Sequential(*model.features, nn.ReLU(inplace=True))
        num_features = model.classifier.in_features
        return backbone, num_features
