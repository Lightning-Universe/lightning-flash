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

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _SEGMENTATION_MODELS_AVAILABLE
from flash.core.utilities.providers import _SEGMENTATION_MODELS

if _SEGMENTATION_MODELS_AVAILABLE:
    import segmentation_models_pytorch as smp

SEMANTIC_SEGMENTATION_BACKBONES = FlashRegistry("backbones")

if _SEGMENTATION_MODELS_AVAILABLE:

    ENCODERS = smp.encoders.get_encoder_names()

    def _load_smp_backbone(backbone: str, **_) -> str:
        return backbone

    for encoder_name in ENCODERS:
        short_name = encoder_name
        if short_name.startswith("timm-"):
            short_name = encoder_name[5:]

        available_weights = smp.encoders.encoders[encoder_name]["pretrained_settings"].keys()
        SEMANTIC_SEGMENTATION_BACKBONES(
            partial(_load_smp_backbone, backbone=encoder_name),
            name=short_name,
            namespace="image/segmentation",
            weights_paths=available_weights,
            providers=_SEGMENTATION_MODELS,
        )
