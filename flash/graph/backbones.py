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

AUTOENCODER_MODELS = ["arga", "argva", "gae", " vgae"]
ENCODER_MODELS = ["DeepGraphInfomax"]
OTHER_MODELS = ["RENet", "SignedGCN", "SchNet", "TGNMemory", "LabelPropagation"]
POSTPROCESSING_MODELS = ["correct_and_smooth"]
RESIDUAL_MODELS = ["deep_GCN_layer", "JumpingKnowledge"]
META_MODELS = ["GNN_explainer"]
EMBEDING_MODELS = ["MetaPath2Vec", "Path2Vec"]

CLASSIFICATION_MODELS = ["AttentiveFP", "GraphUNet", "DimeNet"]

# MODELS


@GRAPH_CLASSIFICATION_BACKBONES(name="AttentiveFP", namespace="graph/classification")  #todo: how to add more tasks?
def load_AttentiveFP(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    edge_dim: int,
    num_layers: int,
    num_timesteps: int,
    dropout: float = 0.0
):
    """Attentive FP backbone from torch geometric"""
    return models.AttentiveFP(in_channels, hidden_channels, out_channels, edge_dim, num_layers, num_timesteps, dropout)


@GRAPH_CLASSIFICATION_BACKBONES(name="DimeNet", namespace="graph/classification")  #todo: how to add more tasks?
def load_DimeNet(
    hidden_channels,
    out_channels,
    num_blocks,
    num_bilinear,
    num_spherical,
    num_radial,
    cutoff=5.0,
    envelope_exponent=5,
    num_before_skip=1,
    num_after_skip=2,
    num_output_layers=3,
    act=ReLU
):
    """DimeNet backbone from torch geometric"""
    return models.DimeNet(
        hidden_channels, out_channels, num_blocks, num_bilinear, num_spherical, num_radial, cutoff, envelope_exponent,
        num_before_skip, num_after_skip, num_output_layers, act
    )


@GRAPH_CLASSIFICATION_BACKBONES(name="GraphUNet", namespace="graph/classification")  #todo: how to add more tasks?
def load_GraphUNet(in_channels, hidden_channels, out_channels, depth, pool_ratios=0.5, sum_res=True, act=ReLU):
    """GraphUNet backbone from torch geometric"""
    return models.GraphUNet(in_channels, hidden_channels, out_channels, depth, pool_ratios, sum_res, act)


## OTHER MODELS
@GRAPH_CLASSIFICATION_BACKBONES(name="RENet", namespace="graph/classification")  #todo: how to add more tasks?
def load_RENet(num_nodes, num_rels, hidden_channels, seq_len, num_layers=1, dropout=0.0, bias=True):
    """RENet backbone from torch geometric"""
    return models.RENet(num_nodes, num_rels, hidden_channels, seq_len, num_layers, dropout, bias)


@GRAPH_CLASSIFICATION_BACKBONES(name="SchNet", namespace="graph/classification")  #todo: how to add more tasks?
def load_SchNet(
    hidden_channels=128,
    num_filters=128,
    num_interactions=6,
    num_gaussians=50,
    cutoff=10.0,
    readout='add',
    dipole=False,
    mean=None,
    std=None,
    atomref=None
):
    """SchNet backbone from torch geometric"""
    return models.SchNet(
        hidden_channels, num_filters, num_interactions, num_gaussians, cutoff, readout, dipole, mean, std, atomref
    )


@GRAPH_CLASSIFICATION_BACKBONES(name="SignedGCM", namespace="graph/classification")  #todo: how to add more tasks?
def load_SignedGCN(in_channels, hidden_channels, num_layers, lamb=5, bias=True):
    """SignedGCN backbone from torch geometric"""
    return models.SignedGCN(in_channels, hidden_channels, num_layers, lamb, bias)


'''@GRAPH_CLASSIFICATION_BACKBONES(name="LabelPropagation", namespace="graph/classification") #todo: how to add more tasks?
def load_LabelPropagation(num_layers: int, alpha: float):
    """LabelPropagation backbone from torch geometric"""
    return models.LabelPropagation(num_layers, alpha)'''

# AUTOENCODER MODELS


@GRAPH_CLASSIFICATION_BACKBONES(name="ARGVA", namespace="graph/classification")  #todo: how to add more tasks?
def load_ARGVA(encoder, discriminator, decoder=None):
    """ARGVA backbone from torch geometric"""
    return models.ARGVA(encoder, discriminator, decoder)


@GRAPH_CLASSIFICATION_BACKBONES(name="ARGA", namespace="graph/classification")  #todo: how to add more tasks?
def load_ARGA(encoder, discriminator, decoder=None):
    """ARGA backbone from torch geometric"""
    return models.ARGA(encoder, discriminator, decoder)


@GRAPH_CLASSIFICATION_BACKBONES(name="GAE", namespace="graph/classification")  #todo: how to add more tasks?
def load_GAE(encoder, decoder=None):
    """GAE backbone from torch geometric"""
    return models.GAE(encoder, decoder)


@GRAPH_CLASSIFICATION_BACKBONES(name="VGAE", namespace="graph/classification")  #todo: how to add more tasks?
def load_VGAE(encoder, decoder=None):
    """VGAE backbone from torch geometric"""
    return models.VGAE(encoder, decoder)
