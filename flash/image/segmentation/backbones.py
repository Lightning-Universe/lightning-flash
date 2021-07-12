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

    ENCODERS = smp.encoders.get_encoder_names()

SEMANTIC_SEGMENTATION_BACKBONES = FlashRegistry("backbones")

if _SEGMENTATION_MODELS_AVAILABLE:

    def _load_smp_model(
        head: Callable,
        backbone: str,
        pretrained: Union[bool, str] = True,
        weights: Optional[str] = None,
        **kwargs
    ) -> Callable:

        if pretrained and weights is not None:
            raise UserWarning("can't set both pretrained and weights!")

        if pretrained:
            weights = "imagenet"

        return head(
            encoder_name=backbone,
            encoder_weights=weights,
            **kwargs,
        )

    for encoder_name in ENCODERS:
        SEMANTIC_SEGMENTATION_BACKBONES(
            partial(_load_smp_model, backbone=encoder_name), name=encoder_name, namespace="image/segmentation"
        )
