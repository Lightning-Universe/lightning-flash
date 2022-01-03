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
import numpy as np
import pytest

from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _SKLEARN_AVAILABLE
from flash.template.classification.data import TemplateData

if _SKLEARN_AVAILABLE:
    from sklearn import datasets


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="sklearn isn't installed")
class TestTemplateData:
    """Tests ``TemplateData``."""

    num_classes: int = 3
    num_features: int = 4

    @staticmethod
    def test_smoke():
        """A simple test that the class can be instantiated."""
        dm = TemplateData(batch_size=2)
        assert dm is not None

    def test_from_numpy(self):
        """Tests that ``TemplateData`` is properly created when using the ``from_numpy`` method."""
        data = np.random.rand(10, self.num_features)
        targets = np.random.randint(0, self.num_classes, (10,))

        # instantiate the data module
        dm = TemplateData.from_numpy(
            train_data=data,
            train_targets=targets,
            val_data=data,
            val_targets=targets,
            test_data=data,
            test_targets=targets,
            batch_size=2,
            num_workers=0,
        )
        assert dm is not None
        assert dm.train_dataloader() is not None
        assert dm.val_dataloader() is not None
        assert dm.test_dataloader() is not None

        # check training data
        data = next(iter(dm.train_dataloader()))
        rows, targets = data[DataKeys.INPUT], data[DataKeys.TARGET]
        assert rows.shape == (2, self.num_features)
        assert targets.shape == (2,)

        # check val data
        data = next(iter(dm.val_dataloader()))
        rows, targets = data[DataKeys.INPUT], data[DataKeys.TARGET]
        assert rows.shape == (2, self.num_features)
        assert targets.shape == (2,)

        # check test data
        data = next(iter(dm.test_dataloader()))
        rows, targets = data[DataKeys.INPUT], data[DataKeys.TARGET]
        assert rows.shape == (2, self.num_features)
        assert targets.shape == (2,)

    @staticmethod
    def test_from_sklearn():
        """Tests that ``TemplateData`` is properly created when using the ``from_sklearn`` method."""
        data = datasets.load_iris()

        # instantiate the data module
        dm = TemplateData.from_sklearn(
            train_bunch=data,
            val_bunch=data,
            test_bunch=data,
            batch_size=2,
            num_workers=0,
        )
        assert dm is not None
        assert dm.train_dataloader() is not None
        assert dm.val_dataloader() is not None
        assert dm.test_dataloader() is not None

        # check training data
        data = next(iter(dm.train_dataloader()))
        rows, targets = data[DataKeys.INPUT], data[DataKeys.TARGET]
        assert rows.shape == (2, dm.num_features)
        assert targets.shape == (2,)

        # check val data
        data = next(iter(dm.val_dataloader()))
        rows, targets = data[DataKeys.INPUT], data[DataKeys.TARGET]
        assert rows.shape == (2, dm.num_features)
        assert targets.shape == (2,)

        # check test data
        data = next(iter(dm.test_dataloader()))
        rows, targets = data[DataKeys.INPUT], data[DataKeys.TARGET]
        assert rows.shape == (2, dm.num_features)
        assert targets.shape == (2,)
