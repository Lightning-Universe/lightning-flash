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
import pytest
import torch
import yaml

from flash.core.data.transforms import merge_transforms
from flash.core.utilities.imports import _GRAPH_AVAILABLE
from flash.graph.classification.data import GraphClassificationData, GraphClassificationPreprocess
from tests.helpers.utils import _GRAPH_TESTING

if _GRAPH_AVAILABLE:
    import networkx as nx
    from networkx.readwrite import (
        adjacency_data,
        cytoscape_data,
        jit_data,
        node_link_data,
        tree_data,
        write_adjlist,
        write_edgelist,
        write_gexf,
        write_gml,
        write_gpickle,
        write_graphml,
        write_pajek,
    )
    from networkx.readwrite.nx_shp import write_shp
    from torch_geometric.data.data import Data as PyGData
    from torch_geometric.datasets import TUDataset
    from torch_geometric.transforms import OneHotDegree
    from torch_geometric.utils import from_networkx


@pytest.mark.skipif(not _GRAPH_TESTING, reason="graph libraries aren't installed.")
class TestGraphClassificationPreprocess:
    """Tests ``GraphClassificationPreprocess``."""

    def test_smoke(self):
        """A simple test that the class can be instantiated."""
        prep = GraphClassificationPreprocess()
        assert prep is not None


@pytest.mark.skipif(not _GRAPH_TESTING, reason="graph libraries aren't installed.")
class TestGraphClassificationData:
    """Tests ``GraphClassificationData``."""

    def test_smoke(self):
        dm = GraphClassificationData()
        assert dm is not None

    def test_from_datasets(self, tmpdir):
        tudataset = TUDataset(root=tmpdir, name="KKI")
        train_dataset = tudataset
        val_dataset = tudataset
        test_dataset = tudataset
        predict_dataset = tudataset

        # instantiate the data module
        dm = GraphClassificationData.from_datasets(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            predict_dataset=predict_dataset,
            train_transform=None,
            val_transform=None,
            test_transform=None,
            predict_transform=None,
            batch_size=2,
        )
        assert dm is not None
        assert dm.train_dataloader() is not None
        assert dm.val_dataloader() is not None
        assert dm.test_dataloader() is not None

        # check training data
        data = next(iter(dm.train_dataloader()))
        assert list(data.x.size())[1] == tudataset.num_features
        assert list(data.y.size()) == [2]

        # check val data
        data = next(iter(dm.val_dataloader()))
        assert list(data.x.size())[1] == tudataset.num_features
        assert list(data.y.size()) == [2]

        # check test data
        data = next(iter(dm.test_dataloader()))
        assert list(data.x.size())[1] == tudataset.num_features
        assert list(data.y.size()) == [2]

    def test_transforms(self, tmpdir):
        tudataset = TUDataset(root=tmpdir, name="KKI")
        train_dataset = tudataset
        val_dataset = tudataset
        test_dataset = tudataset
        predict_dataset = tudataset

        # instantiate the data module
        dm = GraphClassificationData.from_datasets(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            predict_dataset=predict_dataset,
            train_transform=merge_transforms(
                GraphClassificationPreprocess.default_transforms(),
                {"pre_tensor_transform": OneHotDegree(tudataset.num_features - 1)},
            ),
            val_transform=merge_transforms(
                GraphClassificationPreprocess.default_transforms(),
                {"pre_tensor_transform": OneHotDegree(tudataset.num_features - 1)},
            ),
            test_transform=merge_transforms(
                GraphClassificationPreprocess.default_transforms(),
                {"pre_tensor_transform": OneHotDegree(tudataset.num_features - 1)},
            ),
            predict_transform=merge_transforms(
                GraphClassificationPreprocess.default_transforms(),
                {"pre_tensor_transform": OneHotDegree(tudataset.num_features - 1)},
            ),
            batch_size=2,
        )
        assert dm is not None
        assert dm.train_dataloader() is not None
        assert dm.val_dataloader() is not None
        assert dm.test_dataloader() is not None

        # check training data
        data = next(iter(dm.train_dataloader()))
        assert list(data.x.size())[1] == tudataset.num_features * 2
        assert list(data.y.size()) == [2]

        # check val data
        data = next(iter(dm.val_dataloader()))
        assert list(data.x.size())[1] == tudataset.num_features * 2
        assert list(data.y.size()) == [2]

        # check test data
        data = next(iter(dm.test_dataloader()))
        assert list(data.x.size())[1] == tudataset.num_features * 2
        assert list(data.y.size()) == [2]

    def test_from_folder(self, tmpdir):
        G = nx.karate_club_graph()

        write_adjlist(G, tmpdir / 'data.adjlist')
        GraphClassificationData.from_folders(train_folder=tmpdir)

        write_edgelist(G, tmpdir / 'data.edgelist')
        GraphClassificationData.from_folders(train_folder=tmpdir)

        write_gexf(G, tmpdir / 'data.gexf')
        GraphClassificationData.from_folders(train_folder=tmpdir)

        write_gml(G, tmpdir / 'data.gml')
        GraphClassificationData.from_folders(train_folder=tmpdir)

        write_graphml(G, tmpdir / 'data.graphml')
        GraphClassificationData.from_folders(train_folder=tmpdir)

        write_gpickle(G, tmpdir / 'data.gpickle')
        GraphClassificationData.from_folders(train_folder=tmpdir)

        write_pajek(G, tmpdir / 'data.net')
        GraphClassificationData.from_folders(train_folder=tmpdir)

        write_shp(G, tmpdir / 'data.shp')
        GraphClassificationData.from_folders(train_folder=tmpdir)

        yaml.dump(G, tmpdir / 'data.yaml')
        GraphClassificationData.from_folders(train_folder=tmpdir)

        node_link_data(G, tmpdir / 'data.json')
        GraphClassificationData.from_folders(train_folder=tmpdir, json_data_type='node_link')

        adjacency_data(G, tmpdir / 'data.json')
        GraphClassificationData.from_folders(train_folder=tmpdir, json_data_type='adjacency')

        cytoscape_data(G, tmpdir / 'data.json')
        GraphClassificationData.from_folders(train_folder=tmpdir, json_data_type='cytoscape')

        tree_data(G, tmpdir / 'data.json')
        GraphClassificationData.from_folders(train_folder=tmpdir, json_data_type='tree')

        jit_data(G, tmpdir / 'data.json')
        GraphClassificationData.from_folders(train_folder=tmpdir, json_data_type='jit')

    def test_from_data_sequence(self):
        G = nx.karate_club_graph()
        data = from_networkx(G)
        data_x_list = []
        for key in list(G.nodes(data=True))[0][1].keys():
            data_x_list.append(data.__dict__[key])
        data.x = torch.cat(data_x_list, dim=-1)
        data_list = [data, data, data]
        GraphClassificationData.from_data_sequence(data_list)
