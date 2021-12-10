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

import pytest

from flash import DataKeys
from flash.core.utilities.imports import _GRAPH_AVAILABLE
from flash.graph.classification.data import GraphClassificationData
from flash.graph.classification.input_transform import GraphClassificationInputTransform, PyGTransformAdapter
from tests.helpers.utils import _GRAPH_TESTING

if _GRAPH_AVAILABLE:
    from torch_geometric.datasets import TUDataset
    from torch_geometric.transforms import OneHotDegree


@pytest.mark.skipif(not _GRAPH_TESTING, reason="graph libraries aren't installed.")
class TestGraphClassificationData:
    """Tests ``GraphClassificationData``."""

    def test_smoke(self):
        dm = GraphClassificationData(batch_size=2)
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
        data = next(iter(dm.train_dataloader()))[DataKeys.INPUT]
        assert list(data.x.size())[1] == tudataset.num_features
        assert list(data.y.size()) == [2]

        # check val data
        data = next(iter(dm.val_dataloader()))[DataKeys.INPUT]
        assert list(data.x.size())[1] == tudataset.num_features
        assert list(data.y.size()) == [2]

        # check test data
        data = next(iter(dm.test_dataloader()))[DataKeys.INPUT]
        assert list(data.x.size())[1] == tudataset.num_features
        assert list(data.y.size()) == [2]

    def test_transforms(self, tmpdir):
        tudataset = TUDataset(root=tmpdir, name="KKI")
        train_dataset = tudataset
        val_dataset = tudataset
        test_dataset = tudataset
        predict_dataset = tudataset

        class TestInputTransform(GraphClassificationInputTransform):
            @staticmethod
            def _compose(*functions):
                return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

            def per_sample_transform(self):
                return self._compose(
                    super().per_sample_transform(), PyGTransformAdapter(OneHotDegree(tudataset.num_features - 1))
                )

        # instantiate the data module
        dm = GraphClassificationData.from_datasets(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            predict_dataset=predict_dataset,
            train_transform=TestInputTransform,
            val_transform=TestInputTransform,
            test_transform=TestInputTransform,
            predict_transform=TestInputTransform,
            batch_size=2,
        )
        assert dm is not None
        assert dm.train_dataloader() is not None
        assert dm.val_dataloader() is not None
        assert dm.test_dataloader() is not None

        # check training data
        data = next(iter(dm.train_dataloader()))[DataKeys.INPUT]
        assert list(data.x.size())[1] == tudataset.num_features * 2
        assert list(data.y.size()) == [2]

        # check val data
        data = next(iter(dm.val_dataloader()))[DataKeys.INPUT]
        assert list(data.x.size())[1] == tudataset.num_features * 2
        assert list(data.y.size()) == [2]

        # check test data
        data = next(iter(dm.test_dataloader()))[DataKeys.INPUT]
        assert list(data.x.size())[1] == tudataset.num_features * 2
        assert list(data.y.size()) == [2]
