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

import torch.nn as nn
from pytorch_lightning.utilities import rank_zero_warn

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _BOLTS_AVAILABLE, _SEGMENTATION_MODELS_AVAILABLE

if _BOLTS_AVAILABLE:
    if os.getenv("WARN_MISSING_PACKAGE") == "0":
        with warnings.catch_warnings(record=True) as w:
            from pl_bolts.models.vision import UNet
    else:
        from pl_bolts.models.vision import UNet

if _SEGMENTATION_MODELS_AVAILABLE:
    pass

SEMANTIC_SEGMENTATION_HEADS = FlashRegistry("backbones")

if _BOLTS_AVAILABLE:

    def _load_bolts_unet(_, num_classes: int, **kwargs) -> nn.Module:
        rank_zero_warn("The UNet model does not require a backbone, so the backbone will be ignored.", UserWarning)
        return UNet(num_classes, **kwargs)

    SEMANTIC_SEGMENTATION_HEADS(
        fn=_load_bolts_unet, name="unet", namespace="image/segmentation", package="bolts", type="unet"
    )

if _SEGMENTATION_MODELS_AVAILABLE:
    pass
    # def _load_
