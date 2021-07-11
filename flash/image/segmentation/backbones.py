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
from typing import Callable, Optional, Union

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _SEGMENTATION_MODELS_AVAILABLE

if _SEGMENTATION_MODELS_AVAILABLE:
    import segmentation_models_pytorch as smp

    SMP_MODEL_CLASS = [
        smp.Unet, smp.UnetPlusPlus, smp.MAnet, smp.Linknet, smp.FPN, smp.PSPNet, smp.DeepLabV3, smp.DeepLabV3Plus,
        smp.PAN
    ]
    SMP_MODELS = {a.__name__.lower(): a for a in SMP_MODEL_CLASS}

SEMANTIC_SEGMENTATION_BACKBONES = FlashRegistry("backbones")

if _SEGMENTATION_MODELS_AVAILABLE:

    def _load_smp_model(
        model_name: str,
        encoder_name: str,
        pretrained: Union[bool, str] = True,
        in_channels: int = 3,
        num_classes: int = 1,
        weights: Optional[str] = None,
        **kwargs
    ) -> Callable:

        if model_name not in SMP_MODELS:
            raise NotImplementedError(f"{model_name} is not implemented! Supported models -> {SMP_MODELS}")

        if pretrained and weights is not None:
            raise UserWarning("can't set both pretrained and weights!")

        if pretrained:
            weights = "imagenet"

        return smp.create_model(
            arch=model_name,
            encoder_name=encoder_name,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=num_classes,
            **kwargs,
        )

    for model_name in SMP_MODELS:
        SEMANTIC_SEGMENTATION_BACKBONES(
            fn=partial(_load_smp_model, model_name=model_name),
            name=model_name,
            namespace="image/segmentation",
            package="segmentation_models.pytorch"
        )
