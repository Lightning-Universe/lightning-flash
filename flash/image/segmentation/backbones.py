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
import os
import warnings
from functools import partial

import torch.nn as nn
from deprecate import deprecated
from pytorch_lightning.utilities import _BOLTS_AVAILABLE, rank_zero_warn

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _TORCHVISION_AVAILABLE
from flash.image.backbones import catch_url_error

if _TORCHVISION_AVAILABLE:
    from torchvision.models import segmentation

if _BOLTS_AVAILABLE:
    if os.getenv("WARN_MISSING_PACKAGE") == "0":
        with warnings.catch_warnings(record=True) as w:
            from pl_bolts.models.vision import UNet
    else:
        from pl_bolts.models.vision import UNet

FCN_MODELS = ["fcn_resnet50", "fcn_resnet101"]
DEEPLABV3_MODELS = ["deeplabv3_resnet50", "deeplabv3_resnet101", "deeplabv3_mobilenet_v3_large"]
LRASPP_MODELS = ["lraspp_mobilenet_v3_large"]

SEMANTIC_SEGMENTATION_BACKBONES = FlashRegistry("backbones")

if _TORCHVISION_AVAILABLE:

    def _fn_fcn_deeplabv3(model_name: str, num_classes: int, pretrained: bool = True, **kwargs) -> nn.Module:
        model: nn.Module = getattr(segmentation, model_name, None)(pretrained, **kwargs)
        in_channels = model.classifier[-1].in_channels
        model.classifier[-1] = nn.Conv2d(in_channels, num_classes, 1)
        return model

    for model_name in FCN_MODELS + DEEPLABV3_MODELS:
        _type = model_name.split("_")[0]

        SEMANTIC_SEGMENTATION_BACKBONES(
            fn=catch_url_error(partial(_fn_fcn_deeplabv3, model_name)),
            name=model_name,
            namespace="image/segmentation",
            package="torchvision",
            type=_type
        )

    SEMANTIC_SEGMENTATION_BACKBONES(
        fn=deprecated(
            target=None,
            stream=partial(warnings.warn, category=UserWarning),
            deprecated_in="0.3.1",
            remove_in="0.5.0",
            template_mgs="The 'torchvision/fcn_resnet50' backbone has been deprecated since v%(deprecated_in)s in "
            "favor of 'fcn_resnet50'. It will be removed in v%(remove_in)s.",
        )(SEMANTIC_SEGMENTATION_BACKBONES.get("fcn_resnet50")),
        name="torchvision/fcn_resnet50",
    )

    SEMANTIC_SEGMENTATION_BACKBONES(
        fn=deprecated(
            target=None,
            stream=partial(warnings.warn, category=UserWarning),
            deprecated_in="0.3.1",
            remove_in="0.5.0",
            template_mgs="The 'torchvision/fcn_resnet101' backbone has been deprecated since v%(deprecated_in)s in "
            "favor of 'fcn_resnet101'. It will be removed in v%(remove_in)s.",
        )(SEMANTIC_SEGMENTATION_BACKBONES.get("fcn_resnet101")),
        name="torchvision/fcn_resnet101",
    )

    def _fn_lraspp(model_name: str, num_classes: int, pretrained: bool = True, **kwargs) -> nn.Module:
        model: nn.Module = getattr(segmentation, model_name, None)(pretrained, **kwargs)

        low_channels = model.classifier.low_classifier.in_channels
        high_channels = model.classifier.high_classifier.in_channels

        model.classifier.low_classifier = nn.Conv2d(low_channels, num_classes, 1)
        model.classifier.high_classifier = nn.Conv2d(high_channels, num_classes, 1)
        return model

    for model_name in LRASPP_MODELS:
        SEMANTIC_SEGMENTATION_BACKBONES(
            fn=catch_url_error(partial(_fn_lraspp, model_name)),
            name=model_name,
            namespace="image/segmentation",
            package="torchvision",
            type="lraspp"
        )

if _BOLTS_AVAILABLE:

    def load_bolts_unet(num_classes: int, pretrained: bool = False, **kwargs) -> nn.Module:
        if pretrained:
            rank_zero_warn(
                "No pretrained weights are available for the pl_bolts.models.vision.UNet model. This backbone will be "
                "initialized with random weights!", UserWarning
            )
        return UNet(num_classes, **kwargs)

    SEMANTIC_SEGMENTATION_BACKBONES(
        fn=load_bolts_unet, name="unet", namespace="image/segmentation", package="bolts", type="unet"
    )
