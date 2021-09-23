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

if _GRAPH_AVAILABLE:
    import torch_geometric.nn.models as models

GRAPH_BACKBONES = FlashRegistry("backbones")

MODELS = [models.GCN, models.GAT, models.GIN, models.GraphSAGE]
model_names = ["GCN", "GAT", "GIN", "GraphSAGE"]


def _load_graph_backbone(
    model,
    in_channels: int,
    hidden_channels: int = 512,
    num_layers: int = 4,
):
    return model(in_channels, hidden_channels, num_layers)


for model, model_name in zip(MODELS, model_names):
    GRAPH_BACKBONES(name=model_name, namespace="graph")(partial(_load_graph_backbone, model))
