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
import torch.nn as nn

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _VISSL_AVAILABLE

if _VISSL_AVAILABLE:
    from vissl.config.attr_dict import AttrDict
    from vissl.models.model_helpers import RESNET_NORM_LAYER
    from vissl.models.trunks import MODEL_TRUNKS_REGISTRY

    from flash.image.embedding.vissl.adapter import VISSLAdapter
else:
    RESNET_NORM_LAYER = object


def vision_transformer(
    image_size: int = 224,
    patch_size: int = 16,
    hidden_dim: int = 384,
    num_layers: int = 12,
    num_heads: int = 6,
    mlp_dim: int = 1532,
    dropout_rate: float = 0,
    attention_dropout_rate: float = 0,
    drop_path_rate: float = 0,
    qkv_bias: bool = True,
    qk_scale: bool = False,
    classifier: str = "token",
    **kwargs,
) -> nn.Module:

    cfg = VISSLAdapter.get_model_config_template()
    cfg.TRUNK = AttrDict(
        {
            "NAME": "vision_transformer",
            "VISION_TRANSFORMERS": AttrDict(
                {
                    "IMAGE_SIZE": image_size,
                    "PATCH_SIZE": patch_size,
                    "HIDDEN_DIM": hidden_dim,
                    "NUM_LAYERS": num_layers,
                    "NUM_HEADS": num_heads,
                    "MLP_DIM": mlp_dim,
                    "DROPOUT_RATE": dropout_rate,
                    "ATTENTION_DROPOUT_RATE": attention_dropout_rate,
                    "DROP_PATH_RATE": drop_path_rate,
                    "QKV_BIAS": qkv_bias,
                    "QK_SCALE": qk_scale,
                    "CLASSIFIER": classifier,
                }
            ),
        }
    )

    trunk = MODEL_TRUNKS_REGISTRY["vision_transformer"](cfg, model_name="vision_transformer")
    trunk.model_config = cfg

    return trunk, trunk.num_features


def resnet(
    depth: int = 50,
    width_multiplier: int = 1,
    norm: RESNET_NORM_LAYER = None,
    groupnorm_groups: int = 32,
    standardize_convolutions: bool = False,
    groups: int = 1,
    zero_init_residual: bool = False,
    width_per_group: int = 64,
    layer4_stride: int = 2,
    **kwargs,
) -> nn.Module:
    if norm is None:
        norm = RESNET_NORM_LAYER.BatchNorm
    cfg = VISSLAdapter.get_model_config_template()
    cfg.TRUNK = AttrDict(
        {
            "NAME": "resnet",
            "RESNETS": AttrDict(
                {
                    "DEPTH": depth,
                    "WIDTH_MULTIPLIER": width_multiplier,
                    "NORM": norm,
                    "GROUPNORM_GROUPS": groupnorm_groups,
                    "STANDARDIZE_CONVOLUTIONS": standardize_convolutions,
                    "GROUPS": groups,
                    "ZERO_INIT_RESIDUAL": zero_init_residual,
                    "WIDTH_PER_GROUP": width_per_group,
                    "LAYER4_STRIDE": layer4_stride,
                }
            ),
        }
    )

    trunk = MODEL_TRUNKS_REGISTRY["resnet"](cfg, model_name="resnet")
    trunk.model_config = cfg

    return trunk, 2048


def register_vissl_backbones(register: FlashRegistry):
    if _VISSL_AVAILABLE:
        for backbone in (vision_transformer, resnet):
            register(backbone)
