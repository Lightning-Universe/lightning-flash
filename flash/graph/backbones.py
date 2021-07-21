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
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _GRAPH_AVAILABLE

if _GRAPH_AVAILABLE:
    import torch_geometric.nn.models as models

GRAPH_BACKBONES = FlashRegistry("backbones")

MODELS = ["GCN", "GraphSAGE", "GAT", "GIN"]


@GRAPH_BACKBONES(name="GCN", namespace="graph")
def load_GCN(
    in_channels: int,
    hidden_channels: int = 512,
    num_layers: int = 4,
    pretrained: bool = False,
):
    if pretrained:
        raise NotImplementedError('pretrained option for graph backbones not implemented yet')
    """GCN backbone from torch geometric"""
    return models.GCN(in_channels, hidden_channels, num_layers)


@GRAPH_BACKBONES(name="GraphSAGE", namespace="graph")
def load_GraphSAGE(
    in_channels: int,
    hidden_channels: int = 512,
    num_layers: int = 4,
    pretrained: bool = False,
):
    if pretrained:
        raise NotImplementedError('pretrained option for graph backbones not implemented yet')
    """GraphSAGE backbone from torch geometric"""
    return models.GraphSAGE(in_channels, hidden_channels, num_layers)


@GRAPH_BACKBONES(name="GAT", namespace="graph")
def load_GAT(
    in_channels: int,
    hidden_channels: int = 512,
    num_layers: int = 4,
    pretrained: bool = False,
):
    if pretrained:
        raise NotImplementedError('pretrained option for graph backbones not implemented yet')
    """GAT backbone from torch geometric"""
    return models.GAT(in_channels, hidden_channels, num_layers)


@GRAPH_BACKBONES(name="GIN", namespace="graph")
def load_GIN(
    in_channels: int,
    hidden_channels: int = 512,
    num_layers: int = 4,
    pretrained: bool = False,
):
    if pretrained:
        raise NotImplementedError('pretrained option for graph backbones not implemented yet')
    """GIN backbone from torch geometric"""
    return models.GIN(in_channels, hidden_channels, num_layers)
