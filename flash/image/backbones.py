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
import urllib.error
from functools import partial
from typing import Tuple, Union

import torch
from pytorch_lightning.utilities import rank_zero_warn
from torch import nn
from torch.hub import load_state_dict_from_url

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _TIMM_AVAILABLE, _TORCHVISION_AVAILABLE

if _TIMM_AVAILABLE:
    import timm

if _TORCHVISION_AVAILABLE:
    import torchvision
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

MOBILENET_MODELS = ["mobilenet_v2"]
VGG_MODELS = ["vgg11", "vgg13", "vgg16", "vgg19"]
RESNET_MODELS = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x8d"]
DENSENET_MODELS = ["densenet121", "densenet169", "densenet161"]
TORCHVISION_MODELS = MOBILENET_MODELS + VGG_MODELS + RESNET_MODELS + DENSENET_MODELS

IMAGE_CLASSIFIER_BACKBONES = FlashRegistry("backbones")
OBJ_DETECTION_BACKBONES = FlashRegistry("backbones")


def catch_url_error(fn):

    @functools.wraps(fn)
    def wrapper(*args, pretrained=False, **kwargs):
        try:
            return fn(*args, pretrained=pretrained, **kwargs)
        except urllib.error.URLError:
            result = fn(*args, pretrained=False, **kwargs)
            rank_zero_warn(
                "Failed to download pretrained weights for the selected backbone. The backbone has been created with"
                " `pretrained=False` instead. If you are loading from a local checkpoint, this warning can be safely"
                " ignored.", UserWarning
            )
            return result

    return wrapper


if _TORCHVISION_AVAILABLE:

    HTTPS_VISSL = "https://dl.fbaipublicfiles.com/vissl/model_zoo/"
    RESNET50_WEIGHTS_PATHS = {
        "supervised": None,
        "simclr": HTTPS_VISSL + "simclr_rn50_800ep_simclr_8node_resnet_16_07_20.7e8feed1/"
        "model_final_checkpoint_phase799.torch",
        "swav": HTTPS_VISSL + "swav_in1k_rn50_800ep_swav_8node_resnet_27_07_20.a0a6b676/"
        "model_final_checkpoint_phase799.torch",
        "barlow-twins": HTTPS_VISSL + "barlow_twins/barlow_twins_32gpus_4node_imagenet1k_1000ep_resnet50.torch",
    }

    def _fn_mobilenet_vgg(model_name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
        model: nn.Module = getattr(torchvision.models, model_name, None)(pretrained)
        backbone = model.features
        num_features = 512 if model_name in VGG_MODELS else model.classifier[-1].in_features
        return backbone, num_features

    for model_name in MOBILENET_MODELS + VGG_MODELS:
        _type = "mobilenet" if model_name in MOBILENET_MODELS else "vgg"

        IMAGE_CLASSIFIER_BACKBONES(
            fn=catch_url_error(partial(_fn_mobilenet_vgg, model_name)),
            name=model_name,
            namespace="vision",
            package="torchvision",
            type=_type
        )

    def _fn_resnet(model_name: str,
                   pretrained: Union[bool, str] = True,
                   weights_paths: dict = {"supervised": None}) -> Tuple[nn.Module, int]:
        # load according to pretrained if a bool is specified, else set to False
        pretrained_flag = (pretrained and isinstance(pretrained, bool)) or (pretrained == "supervised")

        model: nn.Module = getattr(torchvision.models, model_name, None)(pretrained_flag)
        backbone = nn.Sequential(*list(model.children())[:-2])
        num_features = model.fc.in_features

        model_weights = None
        if not pretrained_flag and isinstance(pretrained, str):
            if pretrained in weights_paths:
                device = next(model.parameters()).get_device()
                model_weights = load_state_dict_from_url(
                    weights_paths[pretrained],
                    map_location=torch.device('cpu') if device == -1 else torch.device(device)
                )

                # add logic here for loading resnet weights from other libraries
                if "classy_state_dict" in model_weights.keys():
                    model_weights = model_weights["classy_state_dict"]["base_model"]["model"]["trunk"]
                    model_weights = {
                        key.replace("_feature_blocks.", "") if "_feature_blocks." in key else key: val
                        for (key, val) in model_weights.items()
                    }
                else:
                    raise KeyError('Unrecognized state dict. Logic for loading the current state dict missing.')
            else:
                raise KeyError(
                    "Requested weights for {0} not available,"
                    " choose from one of {1}".format(model_name, list(weights_paths.keys()))
                )

        if model_weights is not None:
            model.load_state_dict(model_weights, strict=False)

        return backbone, num_features

    def _fn_resnet_fpn(
        model_name: str,
        pretrained: bool = True,
        trainable_layers: bool = True,
        **kwargs,
    ) -> Tuple[nn.Module, int]:
        backbone = resnet_fpn_backbone(model_name, pretrained=pretrained, trainable_layers=trainable_layers, **kwargs)
        return backbone, 256

    for model_name in RESNET_MODELS:
        clf_kwargs = dict(
            fn=catch_url_error(partial(_fn_resnet, model_name=model_name)),
            name=model_name,
            namespace="vision",
            package="torchvision",
            type="resnet",
            weights_paths={"supervised": None}
        )

        if model_name == 'resnet50':
            clf_kwargs.update(
                dict(
                    fn=catch_url_error(
                        partial(_fn_resnet, model_name=model_name, weights_paths=RESNET50_WEIGHTS_PATHS)
                    ),
                    package="multiple",
                    weights_paths=RESNET50_WEIGHTS_PATHS
                )
            )
        IMAGE_CLASSIFIER_BACKBONES(**clf_kwargs)

        OBJ_DETECTION_BACKBONES(
            fn=catch_url_error(partial(_fn_resnet_fpn, model_name)),
            name=model_name,
            package="torchvision",
            type="resnet-fpn"
        )

    def _fn_densenet(model_name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
        model: nn.Module = getattr(torchvision.models, model_name, None)(pretrained)
        backbone = nn.Sequential(*model.features, nn.ReLU(inplace=True))
        num_features = model.classifier.in_features
        return backbone, num_features

    for model_name in DENSENET_MODELS:
        IMAGE_CLASSIFIER_BACKBONES(
            fn=catch_url_error(partial(_fn_densenet, model_name)),
            name=model_name,
            namespace="vision",
            package="torchvision",
            type="densenet"
        )

if _TIMM_AVAILABLE:

    def _fn_timm(
        model_name: str,
        pretrained: bool = True,
        num_classes: int = 0,
        **kwargs,
    ) -> Tuple[nn.Module, int]:
        backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, **kwargs)
        num_features = backbone.num_features
        return backbone, num_features

    for model_name in timm.list_models():

        if model_name in TORCHVISION_MODELS:
            continue

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
