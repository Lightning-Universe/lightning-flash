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
import functools
import urllib.error
from functools import partial
from typing import Tuple, Union

import torch
from pytorch_lightning.utilities import rank_zero_warn
from torch import nn
from torch.hub import load_state_dict_from_url
from torch.nn import ReLU

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _TORCH_GEOMETRIC_AVAILABLE

if _TORCH_GEOMETRIC_AVAILABLE:
    import torch_geometric
    import torch_geometric.nn.models as models

GRAPH_CLASSIFICATION_BACKBONES = FlashRegistry("backbones")

# todo: how to pass arguments to the models:

MODELS = [
    "GCN", "GCNWithJK", "GraphSAGE", "GraphSAGEWithJK", "GAT", "GATWithJK", "GATv2", "GATv2WithJK", "GIN", "GINWithJK",
    "GINE", "GINEWithJK"
]

# MODELS


@GRAPH_CLASSIFICATION_BACKBONES(name="GCN", namespace="graph/classification")  #todo: how to add more tasks?
def load_GCN(
    in_channels: int,
    hidden_channels: int,
    num_layers: int,
):
    """GCN backbone from torch geometric"""
    return models.GCN(in_channels, hidden_channels, num_layers)


@GRAPH_CLASSIFICATION_BACKBONES(name="GCNWithJK", namespace="graph/classification")
def load_GCNWithJK(
    in_channels: int,
    hidden_channels: int,
    num_layers: int,
    num_workers: int,
):
    """GCN backbone with JK from torch geometric"""
    return models.GCNWithJK(in_channels, hidden_channels, num_layers)


@GRAPH_CLASSIFICATION_BACKBONES(name="GraphSAGE", namespace="graph/classification")
def load_GraphSAGE(
    in_channels: int,
    hidden_channels: int,
    num_layers: int,
):
    """GraphSAGE backbone from torch geometric"""
    return models.GraphSAGE(in_channels, hidden_channels, num_layers)


@GRAPH_CLASSIFICATION_BACKBONES(name="GraphSAGEWithJK", namespace="graph/classification")
def load_GraphSAGEWithJK(
    in_channels: int,
    hidden_channels: int,
    num_layers: int,
):
    """GraphSAGE backbone with JK from torch geometric"""
    return models.GraphSAGEWithJK(in_channels, hidden_channels, num_layers)


@GRAPH_CLASSIFICATION_BACKBONES(name="GAT", namespace="graph/classification")
def load_GAT(
    in_channels: int,
    hidden_channels: int,
    num_layers: int,
):
    """GAT backbone from torch geometric"""
    return models.GAT(in_channels, hidden_channels, num_layers)


@GRAPH_CLASSIFICATION_BACKBONES(name="GATWithJK", namespace="graph/classification")
def load_GATWithJK(
    in_channels: int,
    hidden_channels: int,
    num_layers: int,
):
    """GAT backbone with JK from torch geometric"""
    return models.GATWithJK(in_channels, hidden_channels, num_layers)


@GRAPH_CLASSIFICATION_BACKBONES(name="GIN", namespace="graph/classification")
def load_GIN(
    in_channels: int,
    hidden_channels: int,
    num_layers: int,
):
    """GIN backbone from torch geometric"""
    return models.GIN(in_channels, hidden_channels, num_layers)


@GRAPH_CLASSIFICATION_BACKBONES(name="GINWithJK", namespace="graph/classification")
def load_GINWithJK(
    in_channels: int,
    hidden_channels: int,
    num_layers: int,
):
    """GIN backbone with JK from torch geometric"""
    return models.GINWithJK(in_channels, hidden_channels, num_layers)


@GRAPH_CLASSIFICATION_BACKBONES(name="GINE", namespace="graph/classification")
def load_GINE(
    in_channels: int,
    hidden_channels: int,
    num_layers: int,
):
    """GINE backbone from torch geometric"""
    return models.GINE(in_channels, hidden_channels, num_layers)


@GRAPH_CLASSIFICATION_BACKBONES(name="GINEWithJK", namespace="graph/classification")
def load_GINEWithJK(
    in_channels: int,
    hidden_channels: int,
    num_layers: int,
):
    """GINE backbone with JK from torch geometric"""
    return models.GINEWithJK(in_channels, hidden_channels, num_layers)
