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
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _BOLTS_AVAILABLE, _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    from torchvision.models import MobileNetV3, ResNet
    from torchvision.models._utils import IntermediateLayerGetter
    from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3
    from torchvision.models.segmentation.fcn import FCN, FCNHead
    from torchvision.models.segmentation.lraspp import LRASPP

if _BOLTS_AVAILABLE:
    if os.getenv("WARN_MISSING_PACKAGE") == "0":
        with warnings.catch_warnings(record=True) as w:
            from pl_bolts.models.vision import UNet
    else:
        from pl_bolts.models.vision import UNet

RESNET_MODELS = ["resnet50", "resnet101"]
MOBILENET_MODELS = ["mobilenet_v3_large"]

SEMANTIC_SEGMENTATION_HEADS = FlashRegistry("backbones")

if _TORCHVISION_AVAILABLE:

    def _get_backbone_meta(backbone):
        """Adapted from torchvision.models.segmentation.segmentation._segm_model:
        https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/segmentation.py#L25
        """
        if isinstance(backbone, ResNet):
            out_layer = 'layer4'
            out_inplanes = 2048
            aux_layer = 'layer3'
            aux_inplanes = 1024
        elif isinstance(backbone, MobileNetV3):
            backbone = backbone.features
            # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
            # The first and last blocks are always included because they are the C0 (conv1) and Cn.
            stage_indices = [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)]
            stage_indices = [0] + stage_indices + [len(backbone) - 1]
            out_pos = stage_indices[-1]  # use C5 which has output_stride = 16
            out_layer = str(out_pos)
            out_inplanes = backbone[out_pos].out_channels
            aux_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
            aux_layer = str(aux_pos)
            aux_inplanes = backbone[aux_pos].out_channels
        else:
            raise MisconfigurationException(
                f"{type(backbone)} backbone is not currently supported for semantic segmentation."
            )
        return backbone, out_layer, out_inplanes, aux_layer, aux_inplanes

    def _load_fcn_deeplabv3(model_name, backbone, num_classes):
        backbone, out_layer, out_inplanes, aux_layer, aux_inplanes = _get_backbone_meta(backbone)

        return_layers = {out_layer: 'out'}
        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        model_map = {
            "deeplabv3": (DeepLabHead, DeepLabV3),
            "fcn": (FCNHead, FCN),
        }
        classifier = model_map[model_name][0](out_inplanes, num_classes)
        base_model = model_map[model_name][1]

        return base_model(backbone, classifier, None)

    for model_name in ["fcn", "deeplabv3"]:
        SEMANTIC_SEGMENTATION_HEADS(
            fn=partial(_load_fcn_deeplabv3, model_name),
            name=model_name,
            namespace="image/segmentation",
            package="torchvision",
        )

    def _load_lraspp(backbone, num_classes):
        backbone, high_pos, high_channels, low_pos, low_channels = _get_backbone_meta(backbone)
        backbone = IntermediateLayerGetter(backbone, return_layers={low_pos: 'low', high_pos: 'high'})
        return LRASPP(backbone, low_channels, high_channels, num_classes)

    SEMANTIC_SEGMENTATION_HEADS(
        fn=_load_lraspp,
        name="lraspp",
        namespace="image/segmentation",
        package="torchvision",
    )

if _BOLTS_AVAILABLE:

    def _load_bolts_unet(_, num_classes: int, **kwargs) -> nn.Module:
        rank_zero_warn("The UNet model does not require a backbone, so the backbone will be ignored.", UserWarning)
        return UNet(num_classes, **kwargs)

    SEMANTIC_SEGMENTATION_HEADS(
        fn=_load_bolts_unet, name="unet", namespace="image/segmentation", package="bolts", type="unet"
    )
