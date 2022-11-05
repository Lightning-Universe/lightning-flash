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
from typing import Any, Dict, List

from torch.utils.data.dataloader import default_collate

from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _GRAPH_AVAILABLE

if _GRAPH_AVAILABLE:
    from torch_geometric.data import Batch


def _pyg_collate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Helper to collate PyTorch Geometric ``Data`` objects into PyTorch Geometric ``Batch`` objects whilst
    preserving our dictionary format."""
    inputs = Batch.from_data_list([sample[DataKeys.INPUT] for sample in samples])
    if DataKeys.TARGET in samples[0]:
        targets = default_collate([sample[DataKeys.TARGET] for sample in samples])
        return {DataKeys.INPUT: inputs, DataKeys.TARGET: targets}
    return {DataKeys.INPUT: inputs}
