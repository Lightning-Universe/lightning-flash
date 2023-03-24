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

from torch import nn

from flash.core.registry import FlashRegistry  # noqa: F401

# define ImageClassifier registry
CLASSIFIER_HEADS = FlashRegistry("classifier_heads")


def _load_linear_head(num_features: int, num_classes: int) -> nn.Module:
    """Loads a linear head.

    Args:
        num_features: Number of input features.
        num_classes: Number of output classes.

    Returns:
        nn.Module: Linear head.
    """
    return nn.Linear(num_features, num_classes)


CLASSIFIER_HEADS(
    partial(_load_linear_head),
    name="linear",
)
