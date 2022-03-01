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
from torch import nn

from flash.core.registry import FlashRegistry

TEMPLATE_BACKBONES = FlashRegistry("backbones")


@TEMPLATE_BACKBONES(name="mlp-128", namespace="template/classification")
def load_mlp_128(num_features, **_):
    """A simple MLP backbone with 128 hidden units."""
    return (
        nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(True),
            nn.BatchNorm1d(128),
        ),
        128,
    )


@TEMPLATE_BACKBONES(name="mlp-128-256", namespace="template/classification")
def load_mlp_128_256(num_features, **_):
    """Two layer MLP backbone with 128 and 256 hidden units respectively."""
    return (
        nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
        ),
        256,
    )
