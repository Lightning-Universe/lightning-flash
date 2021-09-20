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
from functools import partial
from typing import Union

from torch import nn

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _SEGMENTATION_MODELS_AVAILABLE
from flash.core.utilities.providers import _SEGMENTATION_MODELS

if _SEGMENTATION_MODELS_AVAILABLE:
    import segmentation_models_pytorch as smp

    SMP_MODEL_CLASS = [
        smp.Unet,
        smp.UnetPlusPlus,
        smp.MAnet,
        smp.Linknet,
        smp.FPN,
        smp.PSPNet,
        smp.DeepLabV3,
        smp.DeepLabV3Plus,
        smp.PAN,
    ]
    SMP_MODELS = {a.__name__.lower(): a for a in SMP_MODEL_CLASS}

SEMANTIC_SEGMENTATION_HEADS = FlashRegistry("backbones")

if _SEGMENTATION_MODELS_AVAILABLE:

    def _load_smp_head(
        head: str,
        backbone: str,
        pretrained: Union[bool, str] = True,
        num_classes: int = 1,
        in_channels: int = 3,
        **kwargs,
    ) -> nn.Module:

        if head not in SMP_MODELS:
            raise NotImplementedError(f"{head} is not implemented! Supported heads -> {SMP_MODELS.keys()}")

        encoder_weights = None
        if isinstance(pretrained, str):
            encoder_weights = pretrained
        elif pretrained:
            encoder_weights = "imagenet"

        return smp.create_model(
            arch=head,
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            classes=num_classes,
            in_channels=in_channels,
            **kwargs,
        )

    for model_name in SMP_MODELS:
        SEMANTIC_SEGMENTATION_HEADS(
            partial(_load_smp_head, head=model_name),
            name=model_name,
            namespace="image/segmentation",
            providers=_SEGMENTATION_MODELS,
        )
