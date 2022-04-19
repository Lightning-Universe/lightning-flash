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
from dataclasses import dataclass
from typing import Any, Callable, Dict

from flash.core.data.io.input import DataKeys
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.utilities.samples import to_sample
from flash.core.utilities.imports import _GRAPH_AVAILABLE
from flash.graph.collate import _pyg_collate

if _GRAPH_AVAILABLE:
    from torch_geometric.data import Data
    from torch_geometric.transforms import NormalizeFeatures
else:
    Data = object


@dataclass
class PyGTransformAdapter:
    """Adapter to enable using ``PyG`` transforms within flash.

    Args:
        transform: Transform to apply.
    """

    transform: Callable[[Data], Data]

    def __call__(self, x: Dict[str, Any]):
        data = x[DataKeys.INPUT]
        data.y = x.get(DataKeys.TARGET, None)
        data = self.transform(data)
        return to_sample((data, data.y))


class GraphClassificationInputTransform(InputTransform):
    def collate(self) -> Callable:
        return _pyg_collate

    def per_sample_transform(self) -> Callable:
        return PyGTransformAdapter(NormalizeFeatures())
