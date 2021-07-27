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

import copy
import json
from typing import Any, Dict, Mapping, Optional, Sequence

from torch.utils.data import Dataset

from flash.core.data.data_source import DatasetDataSource, DefaultDataKeys, PathsDataSource, SequenceDataSource
from flash.core.utilities.imports import _GRAPH_AVAILABLE, requires_extras

_GRAPH_EXTENSIONS = ('.gexf', '.gml', '.gpickle', '.graphml', '.leda', '.yaml', '.net', '.edgelist', '.adjlist')

if _GRAPH_AVAILABLE:
    from networkx.readwrite import (
        adjacency_graph,
        cytoscape_graph,
        jit_graph,
        node_link_graph,
        read_adjlist,
        read_edgelist,
        read_gexf,
        read_gml,
        read_gpickle,
        read_graphml,
        read_leda,
        read_pajek,
        read_yaml,
        tree_graph,
    )
    from torch_geometric.data import Data
    from torch_geometric.data import Dataset as TorchGeometricDataset
    from torch_geometric.utils import from_networkx
else:
    Data = object


class GraphDatasetDataSource(DatasetDataSource):

    @requires_extras("graph")
    def load_data(self, data: Dataset, dataset: Any = None) -> Dataset:
        data = super().load_data(data, dataset=dataset)
        if not self.predicting:
            if isinstance(data, TorchGeometricDataset):
                dataset.num_classes = data.num_classes
                dataset.num_features = data.num_features
        return data

    def load_sample(self, sample: Any, dataset: Optional[Any] = None) -> Mapping[str, Any]:
        if isinstance(sample, Data):
            return sample
        return super().load_sample(sample, dataset=dataset)


class GraphSequenceDataSource(SequenceDataSource[Data]):

    @requires_extras("graph")
    def load_data(self, data: Sequence[Data]) -> Sequence:
        # Converting the PyGDataList to the tuple of sequences that load_data expects:

        # Recover the labels
        data_list_y = [data[i].y for i in range(len(data))]

        # Recover the data
        data_list_x = copy(data)
        for data_list_xi in data_list_x:
            data_list_xi.y = None

        # Create data_list
        data_list = (data_list_x, data_list_y)
        data = super().load_data(data_list)

        return data


class GraphPathsDataSource(PathsDataSource):

    def __init__(self, json_data_type=None):
        super().__init__(extensions=_GRAPH_EXTENSIONS)
        self.json_data_type = json_data_type

    @requires_extras("graph")
    def load_sample(self, sample: Dict[str, Any], dataset: Optional[Any] = None) -> Dict[str, Any]:
        """json_data_type required only if data format is .json"""
        graph_path = sample[DefaultDataKeys.INPUT]
        graph = self.default_loader(graph_path, self.json_data_type)
        sample[DefaultDataKeys.INPUT] = from_networkx(graph)
        sample[DefaultDataKeys.METADATA] = {
            "filepath": graph_path,
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "num_features": len(list(graph.nodes(data=True))[0][1].keys()),
        }
        return sample

    def default_loader(self, path: str) -> Any:
        if path.endswith(".gexf"):
            return read_gexf(path)
        elif path.endswith(".gml"):
            return read_gml(path)
        elif path.endswith(".gpickle"):
            return read_gpickle(path)
        elif path.endswith(".graphml"):
            return read_graphml(path)
        elif path.endswith(".leda"):
            return read_leda(path)
        elif path.endswith(".yaml"):
            return read_yaml(path)
        elif path.endswith(".net"):
            return read_pajek(path)
        elif path.endswith(".edgelist"):
            return read_edgelist(path)
        elif path.endswith(".adjlist"):
            return read_adjlist(path)
        elif path.endswith(".json"):
            data = json.load(open(path))
            if self.json_data_type == "node_link":
                return node_link_graph(data)
            elif self.json_data_type == "adjacency":
                return adjacency_graph(data)
            elif self.json_data_type == "cytoscape":
                return cytoscape_graph(data)
            elif self.json_data_type == "jit":
                return jit_graph(data)
            elif self.json_data_type == "tree":
                return tree_graph(data)
        else:
            raise ValueError("Unknown graph file format: {}".format(path))
