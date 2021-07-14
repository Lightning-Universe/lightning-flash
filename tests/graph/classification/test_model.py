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
import torch

from flash import Trainer
from flash.core.data.data_pipeline import DataPipeline
from flash.core.data.data_source import DefaultDataKeys
from flash.core.utilities.imports import _PYTORCH_GEOMETRIC_AVAILABLE
from flash.graph.classification import GraphClassifier
from flash.graph.classification.data import GraphClassificationPreprocess

if _PYTORCH_GEOMETRIC_AVAILABLE:
    from torch_geometric import datasets

# ======== Mock functions ========


class DummyDataset(torch.utils.data.Dataset):
    """We create one or more ``DummyDataset`` classes to provide random data to the model for testing."""

    num_nodes: int = 10
    num_classes: int = 3
    num_features: int = 4

    def __getitem__(self, index):
        return {
            DefaultDataKeys.INPUT: torch.randn(self.num_nodes, self.num_features),
            DefaultDataKeys.TARGET: torch.randint(self.num_classes - 1, (1, ))[0],
        }

    def __len__(self) -> int:
        return 10


# ==============================


@pytest.mark.skipif(not _PYTORCH_GEOMETRIC_AVAILABLE, reason="pytorch geometric isn't installed")
def test_smoke():
    """A simple test that the class can be instantiated."""
    model = GraphClassifier(num_features=1, num_classes=1)
    assert model is not None


@pytest.mark.skipif(not _PYTORCH_GEOMETRIC_AVAILABLE, reason="pytorch geometric isn't installed")
@pytest.mark.parametrize("num_classes", [4, 8])
@pytest.mark.parametrize("num_features", [4, 32])
@pytest.mark.parametrize("num_nodes", [64, 512])
def test_forward(num_nodes, num_features, num_classes):
    """Tests that a tensor can be given to the model forward and gives the correct output size."""
    model = GraphClassifier(
        num_features=num_features,
        num_classes=num_classes,
    )
    model.eval()

    input = torch.rand((num_nodes, num_features))

    out = model(input)
    assert out.shape == num_classes


@pytest.mark.skipif(not _PYTORCH_GEOMETRIC_AVAILABLE, reason="pytorch geometric isn't installed")
def test_train(tmpdir):
    """Tests that the model can be trained on our ``DummyDataset``."""
    model = GraphClassifier(num_features=DummyDataset.num_features, num_classes=DummyDataset.num_classes)
    train_dl = torch.utils.data.DataLoader(DummyDataset(), batch_size=4)
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model, train_dl)


@pytest.mark.skipif(not _PYTORCH_GEOMETRIC_AVAILABLE, reason="pytorch geometric isn't installed")
def test_val(tmpdir):
    """Tests that the model can be validated on our ``DummyDataset``."""
    model = GraphClassifier(num_features=DummyDataset.num_features, num_classes=DummyDataset.num_classes)
    val_dl = torch.utils.data.DataLoader(DummyDataset(), batch_size=4)
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.validate(model, val_dl)


@pytest.mark.skipif(not _PYTORCH_GEOMETRIC_AVAILABLE, reason="pytorch geometric isn't installed")
def test_test(tmpdir):
    """Tests that the model can be tested on our ``DummyDataset``."""
    model = GraphClassifier(num_features=DummyDataset.num_features, num_classes=DummyDataset.num_classes)
    test_dl = torch.utils.data.DataLoader(DummyDataset(), batch_size=4)
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.test(model, test_dl)


@pytest.mark.skipif(not _PYTORCH_GEOMETRIC_AVAILABLE, reason="pytorch geometric isn't installed")
def test_predict_dataset():
    """Tests that we can generate predictions from a pytorch geometric dataset."""
    tudataset = datasets.TUDataset(root='tmpdir', name='KKI')
    model = GraphClassifier(num_features=DummyDataset.num_features, num_classes=DummyDataset.num_classes)
    data_pipe = DataPipeline(preprocess=GraphClassificationPreprocess())
    out = model.predict(tudataset, data_source="dataset", data_pipeline=data_pipe)
    assert isinstance(out[0], int)
