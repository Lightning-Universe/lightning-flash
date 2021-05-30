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

from pathlib import Path

import pytest
from torch.functional import Tensor

from flash.core.data.data_source import DefaultDataKeys
from flash.core.utilities.imports import _PYTORCH_GEOMETRIC_AVAILABLE
from flash.graph.classification.data import GraphClassificationData, GraphClassificationPreprocess

if _PYTORCH_GEOMETRIC_AVAILABLE:
    from torch_geometric.data import Dataset, download_url
    from torch_geometric.datasets import TUDataset
    from torch_geometric.transforms import OneHotDegree


@pytest.mark.skipif(not _PYTORCH_GEOMETRIC_AVAILABLE, reason="pytorch geometric isn't installed.")
class TestTemplatePreprocess:
    """Tests ``TemplatePreprocess``."""

    def test_smoke(self):
        """A simple test that the class can be instantiated."""
        prep = GraphClassificationPreprocess()
        assert prep is not None


@pytest.mark.skipif(not _PYTORCH_GEOMETRIC_AVAILABLE, reason="pytorch geometric isn't installed")
class TestGraphClassificationData:
    """Tests ``estGraphClassificationData``."""

    def test_smoke(self):
        """A simple test that the class can be instantiated."""
        dm = GraphClassificationData()
        assert dm is not None

    def test_from_datasets(self, tmpdir):
        tmpdir = Path(tmpdir)
        tudataset = TUDataset(root='tmpdir', name='KKI')
        """Tests that ``TemplateData`` is properly created when using the ``from_dataset`` method."""
        train_dataset = tudataset,
        val_dataset = tudataset,
        test_dataset = tudataset,
        val_dataset = tudataset,
        predict_dataset = tudataset,

        # instantiate the data module
        dm = GraphClassificationData.from_datasets(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            predict_dataset=predict_dataset,
            train_transform=None,
            val_transform=None,
            test_transform=None,
            predict_transform=None
        )
        assert dm is not None
        assert dm.train_dataloader() is not None
        assert dm.val_dataloader() is not None
        assert dm.test_dataloader() is not None

        # check training data
        data = next(iter(dm.train_dataloader()))
        input, targets = data[DefaultDataKeys.INPUT], data[DefaultDataKeys.TARGET]
        assert list(input.size())[1] == tudataset.num_features
        assert list(targets.size()) == [1]

        # check val data
        data = next(iter(dm.val_dataloader()))
        input, targets = data[DefaultDataKeys.INPUT], data[DefaultDataKeys.TARGET]
        assert list(input.size())[1] == tudataset.num_features
        assert list(targets.size()) == [1]

        # check test data
        data = next(iter(dm.test_dataloader()))
        input, targets = data[DefaultDataKeys.INPUT], data[DefaultDataKeys.TARGET]
        assert list(input.size())[1] == tudataset.num_features
        assert list(targets.size()) == [1]

    def test_transforms(self, tmpdir):
        tmpdir = Path(tmpdir)
        tudataset = TUDataset(root='tmpdir', name='KKI')
        """Tests that ``TemplateData`` is properly created when using the ``from_dataset`` method."""
        train_dataset = tudataset,
        val_dataset = tudataset,
        test_dataset = tudataset,
        val_dataset = tudataset,
        predict_dataset = tudataset,

        # instantiate the data module
        dm = GraphClassificationData.from_datasets(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            predict_dataset=predict_dataset,
            train_transform=OneHotDegree,
            val_transform=OneHotDegree,
            test_transform=OneHotDegree,
            predict_transform=OneHotDegree
        )
        assert dm is not None
        assert dm.train_dataloader() is not None
        assert dm.val_dataloader() is not None
        assert dm.test_dataloader() is not None

        # check training data
        data = next(iter(dm.train_dataloader()))
        input, targets = data[DefaultDataKeys.INPUT], data[DefaultDataKeys.TARGET]
        assert list(input.size())[1] == tudataset.num_features
        assert list(targets.size()) == [1]

        # check val data
        data = next(iter(dm.val_dataloader()))
        input, targets = data[DefaultDataKeys.INPUT], data[DefaultDataKeys.TARGET]
        assert list(input.size())[1] == tudataset.num_features
        assert list(targets.size()) == [1]

        # check test data
        data = next(iter(dm.test_dataloader()))
        input, targets = data[DefaultDataKeys.INPUT], data[DefaultDataKeys.TARGET]
        assert list(input.size())[1] == tudataset.num_features
        assert list(targets.size()) == [1]
