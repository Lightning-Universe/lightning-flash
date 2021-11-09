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
from flash.core.utilities.imports import _GRAPH_AVAILABLE
from flash.core.utilities.providers import _PYTORCH_GEOMETRIC

if _GRAPH_AVAILABLE:
    from torch_geometric.nn.models import GAT, GCN, GIN, GraphSAGE

    MODELS = {"GCN": GCN, "GraphSAGE": GraphSAGE, "GAT": GAT, "GIN": GIN}
else:
    MODELS = {}

GRAPH_BACKBONES = FlashRegistry("backbones")


def _load_graph_backbone(
    model_name: str,
    in_channels: int,
    hidden_channels: int = 512,
    num_layers: int = 4,
):
    model = MODELS[model_name]
    return model(in_channels, hidden_channels, num_layers)


for model_name in MODELS.keys():
    GRAPH_BACKBONES(name=model_name, providers=_PYTORCH_GEOMETRIC)(partial(_load_graph_backbone, model_name))
