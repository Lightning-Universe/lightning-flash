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

import torchvision
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn as nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from flash.utils.imports import _BOLTS_AVAILABLE, _TIMM_AVAILABLE

if _TIMM_AVAILABLE:
    import timm

if _BOLTS_AVAILABLE:
    from pl_bolts.models.self_supervised import SimCLR, SwAV

ROOT_S3_BUCKET = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com"

MOBILENET_MODELS = ["mobilenet_v2"]
VGG_MODELS = ["vgg11", "vgg13", "vgg16", "vgg19"]
RESNET_MODELS = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x8d"]
DENSENET_MODELS = ["densenet121", "densenet169", "densenet161"]
TORCHVISION_MODELS = MOBILENET_MODELS + VGG_MODELS + RESNET_MODELS + DENSENET_MODELS

BOLTS_MODELS = ["simclr-imagenet", "swav-imagenet"]


def backbone_and_num_features(
    model_name: str,
    fpn: bool = False,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    **kwargs
) -> Tuple[nn.Module, int]:
    """
    Args:
        model_name: backbone supported by `torchvision` and `bolts`
        fpn: If True, creates a Feature Pyramind Network on top of Resnet based CNNs.
        pretrained: if true, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers: number of trainable resnet layers starting from final block.

    >>> backbone_and_num_features('mobilenet_v2')  # doctest: +ELLIPSIS
    (Sequential(...), 1280)
    >>> backbone_and_num_features('resnet50', fpn=True)  # doctest: +ELLIPSIS
    (BackboneWithFPN(...), 256)
    >>> backbone_and_num_features('swav-imagenet')  # doctest: +ELLIPSIS
    (Sequential(...), 2048)
    """
    if fpn:
        if model_name in RESNET_MODELS:
            backbone = resnet_fpn_backbone(
                model_name, pretrained=pretrained, trainable_layers=trainable_backbone_layers, **kwargs
            )
            fpn_out_channels = 256
            return backbone, fpn_out_channels
        else:
            rank_zero_warn(f"{model_name} backbone is not supported with `fpn=True`, `fpn` won't be added.")

    if model_name in BOLTS_MODELS:
        return bolts_backbone_and_num_features(model_name)

    if model_name in TORCHVISION_MODELS:
        return torchvision_backbone_and_num_features(model_name, pretrained)

    if _TIMM_AVAILABLE and model_name in timm.list_models():
        return timm_backbone_and_num_features(model_name, pretrained)

    raise ValueError(f"{model_name} is not supported yet.")


def bolts_backbone_and_num_features(model_name: str) -> Tuple[nn.Module, int]:
    """
    >>> bolts_backbone_and_num_features('simclr-imagenet')  # doctest: +ELLIPSIS
    (Sequential(...), 2048)
    >>> bolts_backbone_and_num_features('swav-imagenet')  # doctest: +ELLIPSIS
    (Sequential(...), 2048)
    """

    # TODO: maybe we should plain pytorch weights so we don't need to rely on bolts to load these
    # also mabye just use torchhub for the ssl lib
    def load_simclr_imagenet(path_or_url: str = f"{ROOT_S3_BUCKET}/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt"):
        simclr = SimCLR.load_from_checkpoint(path_or_url, strict=False)
        # remove the last two layers & turn it into a Sequential model
        backbone = nn.Sequential(*list(simclr.encoder.children())[:-2])
        return backbone, 2048

    def load_swav_imagenet(path_or_url: str = f"{ROOT_S3_BUCKET}/swav/swav_imagenet/swav_imagenet.pth.tar"):
        swav = SwAV.load_from_checkpoint(path_or_url, strict=True)
        # remove the last two layers & turn it into a Sequential model
        backbone = nn.Sequential(*list(swav.model.children())[:-2])
        return backbone, 2048

    models = {
        'simclr-imagenet': load_simclr_imagenet,
        'swav-imagenet': load_swav_imagenet,
    }
    if not _BOLTS_AVAILABLE:
        raise MisconfigurationException("Bolts isn't installed. Please, use ``pip install lightning-bolts``.")
    if model_name in models:
        return models[model_name]()

    raise ValueError(f"{model_name} is not supported yet.")


def torchvision_backbone_and_num_features(model_name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    >>> torchvision_backbone_and_num_features('mobilenet_v2')  # doctest: +ELLIPSIS
    (Sequential(...), 1280)
    >>> torchvision_backbone_and_num_features('resnet18')  # doctest: +ELLIPSIS
    (Sequential(...), 512)
    >>> torchvision_backbone_and_num_features('densenet121')  # doctest: +ELLIPSIS
    (Sequential(...), 1024)
    """
    model = getattr(torchvision.models, model_name, None)
    if model is None:
        raise MisconfigurationException(f"{model_name} is not supported by torchvision")

    if model_name in MOBILENET_MODELS + VGG_MODELS:
        model = model(pretrained=pretrained)
        backbone = model.features
        num_features = 512 if model_name in VGG_MODELS else model.classifier[-1].in_features
        return backbone, num_features

    elif model_name in RESNET_MODELS:
        model = model(pretrained=pretrained)
        # remove the last two layers & turn it into a Sequential model
        backbone = nn.Sequential(*list(model.children())[:-2])
        num_features = model.fc.in_features
        return backbone, num_features

    elif model_name in DENSENET_MODELS:
        model = model(pretrained=pretrained)
        backbone = nn.Sequential(*model.features, nn.ReLU(inplace=True))
        num_features = model.classifier.in_features
        return backbone, num_features

    raise ValueError(f"{model_name} is not supported yet.")


def timm_backbone_and_num_features(model_name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:

    if model_name in timm.list_models():
        backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')
        num_features = backbone.num_features
        return backbone, num_features

    raise ValueError(
        f"{model_name} is not supported in timm yet. https://rwightman.github.io/pytorch-image-models/models/"
    )
