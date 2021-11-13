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
from unittest import mock

import pytest
import torch

from flash import Trainer
from flash.__main__ import main
from flash.core.data.data_pipeline import DataPipeline
from flash.core.utilities.imports import _GRAPH_AVAILABLE
from flash.graph.classification import GraphClassifier
from flash.graph.classification.data import GraphClassificationInputTransform
from tests.helpers.utils import _GRAPH_TESTING

if _GRAPH_AVAILABLE:
    from torch_geometric import datasets


@pytest.mark.skipif(not _GRAPH_TESTING, reason="pytorch geometric isn't installed")
def test_smoke():
    """A simple test that the class can be instantiated."""
    model = GraphClassifier(num_features=1, num_classes=1)
    assert model is not None


@pytest.mark.skipif(not _GRAPH_TESTING, reason="pytorch geometric isn't installed")
def test_train(tmpdir):
    """Tests that the model can be trained on a pytorch geometric dataset."""
    tudataset = datasets.TUDataset(root=tmpdir, name="KKI")
    model = GraphClassifier(num_features=tudataset.num_features, num_classes=tudataset.num_classes)
    model.data_pipeline = DataPipeline(input_transform=GraphClassificationInputTransform())
    train_dl = torch.utils.data.DataLoader(tudataset, batch_size=4)
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model, train_dl)


@pytest.mark.skipif(not _GRAPH_TESTING, reason="pytorch geometric isn't installed")
def test_val(tmpdir):
    """Tests that the model can be validated on a pytorch geometric dataset."""
    tudataset = datasets.TUDataset(root=tmpdir, name="KKI")
    model = GraphClassifier(num_features=tudataset.num_features, num_classes=tudataset.num_classes)
    model.data_pipeline = DataPipeline(input_transform=GraphClassificationInputTransform())
    val_dl = torch.utils.data.DataLoader(tudataset, batch_size=4)
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.validate(model, val_dl)


@pytest.mark.skipif(not _GRAPH_TESTING, reason="pytorch geometric isn't installed")
def test_test(tmpdir):
    """Tests that the model can be tested on a pytorch geometric dataset."""
    tudataset = datasets.TUDataset(root=tmpdir, name="KKI")
    model = GraphClassifier(num_features=tudataset.num_features, num_classes=tudataset.num_classes)
    model.data_pipeline = DataPipeline(input_transform=GraphClassificationInputTransform())
    test_dl = torch.utils.data.DataLoader(tudataset, batch_size=4)
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.test(model, test_dl)


@pytest.mark.skipif(not _GRAPH_TESTING, reason="pytorch geometric isn't installed")
def test_predict_dataset(tmpdir):
    """Tests that we can generate predictions from a pytorch geometric dataset."""
    tudataset = datasets.TUDataset(root=tmpdir, name="KKI")
    model = GraphClassifier(num_features=tudataset.num_features, num_classes=tudataset.num_classes)
    data_pipe = DataPipeline(input_transform=GraphClassificationInputTransform())
    out = model.predict(tudataset, input="datasets", data_pipeline=data_pipe)
    assert isinstance(out[0], int)


@pytest.mark.skipif(not _GRAPH_TESTING, reason="pytorch geometric isn't installed")
def test_cli():
    cli_args = ["flash", "graph_classification", "--trainer.fast_dev_run", "True"]
    with mock.patch("sys.argv", cli_args):
        try:
            main()
        except SystemExit:
            pass
