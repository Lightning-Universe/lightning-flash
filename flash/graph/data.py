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
from typing import Any, Dict, Optional

import torch

from flash.data.data_source import DataSource, DefaultDataKeys, NumpyDataSource, PathsDataSource, TensorDataSource

GRAPH_EXTENSIONS = ['.pt']
#todo: how to get default_loader and GRAPH_EXTENSIONS? These were provided by torchvision in the case of vision


class GraphPathsDataSource(PathsDataSource):

    def __init__(self):
        super().__init__(extensions=GRAPH_EXTENSIONS)

    def load_sample(self, sample: Dict[str, Any], dataset: Optional[Any] = None) -> Dict[str, Any]:
        # seems to me from https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
        # that PyG uses torch.load()
        default_loader = torch.load  # todo: is this right?
        sample[DefaultDataKeys.INPUT] = default_loader(sample[DefaultDataKeys.INPUT])
        return sample

    # todo: an alternative would be to load from networkx https://networkx.org/documentation/stable/reference/readwrite/index.html recognised files
    # In such case one can use https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html?highlight=networkx%20to%20pyg#torch_geometric.utils.from_networkx
    # to convert it to torch_geometric.data.Data instance
