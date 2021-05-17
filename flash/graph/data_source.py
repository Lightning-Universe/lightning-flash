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
from torch.utils.data import Dataset

from flash.core.data.auto_dataset import AutoDataset
from flash.core.data.data_source import DatasetDataSource, DefaultDataKeys, PathsDataSource
from flash.core.utilities.imports import _PYTORCH_GEOMETRIC_AVAILABLE

if _PYTORCH_GEOMETRIC_AVAILABLE:
    from torch_geometric.data import Dataset as PyGDataset

GRAPH_EXTENSIONS = ['.pt']
#todo: how to get default_loader and GRAPH_EXTENSIONS? These were provided by torchvision in the case of vision


class GraphDatasetSource(DatasetDataSource):

    def load_data(self, dataset: Dataset, auto_dataset: AutoDataset) -> Dataset:
        data = super().load_data(dataset, auto_dataset)
        if self.training:
            if isinstance(dataset, PyGDataset):
                auto_dataset.num_classes = dataset.num_classes
        return data


class GraphPathsDataSource(PathsDataSource):

    def __init__(self):
        super().__init__(extensions=GRAPH_EXTENSIONS)

    def load_sample(self, sample: Dict[str, Any], dataset: Optional[Any] = None) -> Dict[str, Any]:
        # seems to me from https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
        # that PyG uses torch.load()
        default_loader = torch.load  # todo: is this right?
        sample[DefaultDataKeys.INPUT] = default_loader(sample[DefaultDataKeys.INPUT])
        return sample # todo: what fields should sample have here?

    # todo: an alternative would be to load from networkx https://networkx.org/documentation/stable/reference/readwrite/index.html recognised files
    # In such case one can use https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html?highlight=networkx%20to%20pyg#torch_geometric.utils.from_networkx
    # to convert it to torch_geometric.data.Data instance.
