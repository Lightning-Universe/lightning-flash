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
from typing import List, Union

import torch.nn as nn
from torch import Tensor

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _VISSL_AVAILABLE

if _VISSL_AVAILABLE:
    from vissl.config.attr_dict import AttrDict
    from vissl.models.heads import MODEL_HEADS_REGISTRY, register_model_head

    from flash.image.embedding.vissl.adapter import VISSLAdapter
else:
    AttrDict = object


class SimCLRHead(nn.Module):
    """VISSL adpots a complicated config input to create an MLP.

    This class simplifies the standard SimCLR projection head.
    Can be configured to be used with barlow twins as well.

    Returns MLP according to dimensions provided as a list.
    linear-layer -> batch-norm (if flag) -> Relu -> ...

    Args:
        model_config: Model config AttrDict from VISSL
        dims: list of dimensions for creating a projection head
        use_bn: use batch-norm after each linear layer or not
    """

    def __init__(
        self,
        model_config: AttrDict,
        dims: List[int] = [2048, 2048, 128],
        use_bn: bool = True,
        **kwargs,
    ) -> nn.Module:
        super().__init__()

        self.model_config = model_config
        self.dims = dims
        self.use_bn = use_bn

        self.clf = self.create_mlp()

    def create_mlp(self):
        layers = []
        last_dim = self.dims[0]

        for dim in self.dims[1:-1]:
            layers.append(nn.Linear(last_dim, dim))

            if self.use_bn:
                layers.append(
                    nn.BatchNorm1d(
                        dim,
                        eps=self.model_config.HEAD.BATCHNORM_EPS,
                        momentum=self.model_config.HEAD.BATCHNORM_MOMENTUM,
                    )
                )

            layers.append(nn.ReLU(inplace=True))
            last_dim = dim

        layers.append(nn.Linear(last_dim, self.dims[-1]))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.clf(x)


if _VISSL_AVAILABLE:
    SimCLRHead = register_model_head("simclr_head")(SimCLRHead)


def simclr_head(
    num_features: int = 2048,
    embedding_dim: int = 128,
    dims: List[int] = [2048],
    use_bn: bool = True,
    **kwargs,
) -> nn.Module:
    cfg = VISSLAdapter.get_model_config_template()
    head_kwargs = {
        "dims": [num_features] + dims + [embedding_dim],
        "use_bn": use_bn,
    }

    cfg.HEAD.PARAMS.append(["simclr_head", head_kwargs])

    head = MODEL_HEADS_REGISTRY["simclr_head"](cfg, **head_kwargs)
    head.model_config = cfg

    return head


def swav_head(
    num_features: int = 2048,
    embedding_dim: int = 128,
    dims: List[int] = [2048],
    use_bn: bool = True,
    num_clusters: Union[int, List[int]] = [3000],
    use_bias: bool = True,
    return_embeddings: bool = True,
    skip_last_bn: bool = True,
    normalize_feats: bool = True,
    activation_name: str = "ReLU",
    use_weight_norm_prototypes: bool = False,
    **kwargs,
) -> nn.Module:
    cfg = VISSLAdapter.get_model_config_template()
    head_kwargs = {
        "dims": [num_features] + dims + [embedding_dim],
        "use_bn": use_bn,
        "num_clusters": [num_clusters] if isinstance(num_clusters, int) else num_clusters,
        "use_bias": use_bias,
        "return_embeddings": return_embeddings,
        "skip_last_bn": skip_last_bn,
        "normalize_feats": normalize_feats,
        "activation_name": activation_name,
        "use_weight_norm_prototypes": use_weight_norm_prototypes,
    }

    cfg.HEAD.PARAMS.append(["swav_head", head_kwargs])

    head = MODEL_HEADS_REGISTRY["swav_head"](cfg, **head_kwargs)
    head.model_config = cfg

    return head


def barlow_twins_head(
    latent_embedding_dim: int = 8192,
    **kwargs,
) -> nn.Module:
    return simclr_head(embedding_dim=latent_embedding_dim, **kwargs)


def register_vissl_heads(register: FlashRegistry):
    for ssl_head in (swav_head, simclr_head, barlow_twins_head):
        register(ssl_head)
