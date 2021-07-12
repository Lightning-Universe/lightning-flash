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
from typing import Callable

from pytorch_lightning.utilities.exceptions import MisconfigurationException

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _SEGMENTATION_MODELS_AVAILABLE, _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    from torchvision.models import MobileNetV3, ResNet
    from torchvision.models._utils import IntermediateLayerGetter
    from torchvision.models.segmentation.lraspp import LRASPP

if _SEGMENTATION_MODELS_AVAILABLE:
    import segmentation_models_pytorch as smp

    SMP_MODEL_CLASS = [
        smp.Unet, smp.UnetPlusPlus, smp.MAnet, smp.Linknet, smp.FPN, smp.PSPNet, smp.DeepLabV3, smp.DeepLabV3Plus,
        smp.PAN
    ]
    SMP_MODELS = {a.__name__.lower(): a for a in SMP_MODEL_CLASS}

SEMANTIC_SEGMENTATION_HEADS = FlashRegistry("backbones")

if _SEGMENTATION_MODELS_AVAILABLE:

    def _load_smp_head(
        head: str,
        backbone: str,
        pretrained: bool = True,
        num_classes: int = 1,
        in_channels: int = 3,
        **kwargs,
    ) -> Callable:

        if head not in SMP_MODELS:
            raise NotImplementedError(f"{head} is not implemented! Supported heads -> {SMP_MODELS.keys()}")

        encoder_weights = None
        if pretrained:
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
            package="segmentation_models.pytorch"
        )

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
