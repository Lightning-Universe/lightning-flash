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
    from vissl.models.trunks import MODEL_TRUNKS_REGISTRY

    from flash.image.embedding.vissl.adapter import VISSLAdapter


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
                    "image_size": image_size,
                    "patch_size": patch_size,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "mlp_dim": mlp_dim,
                    "dropout_rate": dropout_rate,
                    "attention_dropout_rate": attention_dropout_rate,
                    "drop_path_rate": drop_path_rate,
                    "qkv_bias": qkv_bias,
                    "qk_scale": qk_scale,
                    "classifier": classifier,
                }
            ),
        }
    )

    trunk = MODEL_TRUNKS_REGISTRY["vision_transformer"](cfg, model_name="vision_transformer")
    trunk.model_config = cfg

    return trunk, trunk.num_features


def register_vissl_backbones(register: FlashRegistry):
    register(vision_transformer)
