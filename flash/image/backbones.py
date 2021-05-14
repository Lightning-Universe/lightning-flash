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
import functools
import os
import urllib.error
import warnings
from functools import partial
from typing import Tuple

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import _BOLTS_AVAILABLE, rank_zero_warn
from torch import nn as nn

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _IMAGE_AVAILABLE, _TIMM_AVAILABLE, _TORCHVISION_AVAILABLE

if _TIMM_AVAILABLE:
    import timm

if _TORCHVISION_AVAILABLE:
    import torchvision
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

if _BOLTS_AVAILABLE:
    if os.getenv("WARN_MISSING_PACKAGE") == "0":
        with warnings.catch_warnings(record=True) as w:
            from pl_bolts.models.self_supervised import SimCLR, SwAV
    else:
        from pl_bolts.models.self_supervised import SimCLR, SwAV

ROOT_S3_BUCKET = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com"

MOBILENET_MODELS = ["mobilenet_v2"]
VGG_MODELS = ["vgg11", "vgg13", "vgg16", "vgg19"]
RESNET_MODELS = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x8d"]
DENSENET_MODELS = ["densenet121", "densenet169", "densenet161"]
TORCHVISION_MODELS = MOBILENET_MODELS + VGG_MODELS + RESNET_MODELS + DENSENET_MODELS
BOLTS_MODELS = ["simclr-imagenet", "swav-imagenet"]

IMAGE_CLASSIFIER_BACKBONES = FlashRegistry("backbones")
OBJ_DETECTION_BACKBONES = FlashRegistry("backbones")


def catch_url_error(fn):

    @functools.wraps(fn)
    def wrapper(pretrained=False, **kwargs):
        try:
            return fn(pretrained=pretrained, **kwargs)
        except urllib.error.URLError:
            result = fn(pretrained=False, **kwargs)
            rank_zero_warn(
                "Failed to download pretrained weights for the selected backbone. The backbone has been created with"
                " `pretrained=False` instead. If you are loading from a local checkpoint, this warning can be safely"
                " ignored.", UserWarning
            )
            return result

    return wrapper


@IMAGE_CLASSIFIER_BACKBONES(name="simclr-imagenet", namespace="vision", package="bolts")
def load_simclr_imagenet(path_or_url: str = f"{ROOT_S3_BUCKET}/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt", **_):
    simclr: LightningModule = SimCLR.load_from_checkpoint(path_or_url, strict=False)
    # remove the last two layers & turn it into a Sequential model
    backbone = nn.Sequential(*list(simclr.encoder.children())[:-2])
    return backbone, 2048


@IMAGE_CLASSIFIER_BACKBONES(name="swav-imagenet", namespace="vision", package="bolts")
def load_swav_imagenet(
    path_or_url: str = f"{ROOT_S3_BUCKET}/swav/swav_imagenet/swav_imagenet.pth.tar",
    **_,
) -> Tuple[nn.Module, int]:
    swav: LightningModule = SwAV.load_from_checkpoint(path_or_url, strict=True)
    # remove the last two layers & turn it into a Sequential model
    backbone = nn.Sequential(*list(swav.model.children())[:-2])
    return backbone, 2048


if _TORCHVISION_AVAILABLE:

    for model_name in MOBILENET_MODELS + VGG_MODELS:

        def _fn_mobilenet_vgg(model_name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
            model: nn.Module = getattr(torchvision.models, model_name, None)(pretrained)
            backbone = model.features
            num_features = 512 if model_name in VGG_MODELS else model.classifier[-1].in_features
            return backbone, num_features

        _type = "mobilenet" if model_name in MOBILENET_MODELS else "vgg"

        IMAGE_CLASSIFIER_BACKBONES(
            fn=catch_url_error(partial(_fn_mobilenet_vgg, model_name)),
            name=model_name,
            namespace="vision",
            package="torchvision",
            type=_type
        )

    for model_name in RESNET_MODELS:

        def _fn_resnet(model_name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
            model: nn.Module = getattr(torchvision.models, model_name, None)(pretrained)
            backbone = nn.Sequential(*list(model.children())[:-2])
            num_features = model.fc.in_features
            return backbone, num_features

        IMAGE_CLASSIFIER_BACKBONES(
            fn=catch_url_error(partial(_fn_resnet, model_name)),
            name=model_name,
            namespace="vision",
            package="torchvision",
            type="resnet"
        )

        def _fn_resnet_fpn(
            model_name: str,
            pretrained: bool = True,
            trainable_layers: bool = True,
            **kwargs,
        ) -> Tuple[nn.Module, int]:
            backbone = resnet_fpn_backbone(
                model_name, pretrained=pretrained, trainable_layers=trainable_layers, **kwargs
            )
            return backbone, 256

        OBJ_DETECTION_BACKBONES(
            fn=catch_url_error(partial(_fn_resnet_fpn, model_name)),
            name=model_name,
            package="torchvision",
            type="resnet-fpn"
        )

    for model_name in DENSENET_MODELS:

        def _fn_densenet(model_name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
            model: nn.Module = getattr(torchvision.models, model_name, None)(pretrained)
            backbone = nn.Sequential(*model.features, nn.ReLU(inplace=True))
            num_features = model.classifier.in_features
            return backbone, num_features

        IMAGE_CLASSIFIER_BACKBONES(
            fn=catch_url_error(partial(_fn_densenet, model_name)),
            name=model_name,
            namespace="vision",
            package="torchvision",
            type="densenet"
        )

if _TIMM_AVAILABLE:
    for model_name in timm.list_models():

        if model_name in TORCHVISION_MODELS:
            continue

        def _fn_timm(
            model_name: str,
            pretrained: bool = True,
            num_classes: int = 0,
            global_pool: str = '',
        ) -> Tuple[nn.Module, int]:
            backbone = timm.create_model(
                model_name, pretrained=pretrained, num_classes=num_classes, global_pool=global_pool
            )
            num_features = backbone.num_features
            return backbone, num_features

        IMAGE_CLASSIFIER_BACKBONES(
            fn=catch_url_error(partial(_fn_timm, model_name)), name=model_name, namespace="vision", package="timm"
        )


# Paper: Emerging Properties in Self-Supervised Vision Transformers
# https://arxiv.org/abs/2104.14294 from Mathilde Caron and al. (29 Apr 2021)
# weights from https://github.com/facebookresearch/dino
def dino_deits16(*_, **__):
    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_deits16')
    return backbone, 384


def dino_deits8(*_, **__):
    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_deits8')
    return backbone, 384


def dino_vitb16(*_, **__):
    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    return backbone, 768


def dino_vitb8(*_, **__):
    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
    return backbone, 768


IMAGE_CLASSIFIER_BACKBONES(dino_deits16)
IMAGE_CLASSIFIER_BACKBONES(dino_deits8)
IMAGE_CLASSIFIER_BACKBONES(dino_vitb16)
IMAGE_CLASSIFIER_BACKBONES(dino_vitb8)
