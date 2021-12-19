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

from flash import RunningStage, Trainer
from flash.__main__ import main
from flash.core.data.data_module import DataModule
from flash.core.utilities.imports import _GRAPH_AVAILABLE
from flash.graph.classification.input import GraphClassificationDatasetInput
from flash.graph.classification.input_transform import GraphClassificationInputTransform
from flash.graph.node_classification import GraphNodeClassifier
from tests.helpers.utils import _GRAPH_TESTING

if _GRAPH_AVAILABLE:
    from torch_geometric import datasets


@pytest.mark.skipif(not _GRAPH_TESTING, reason="pytorch geometric isn't installed")
def test_smoke():
    """A simple test that the class can be instantiated."""
    model = GraphNodeClassifier(num_features=1, num_classes=1)
    assert model is not None


@pytest.mark.skipif(not _GRAPH_TESTING, reason="pytorch geometric isn't installed")
def test_train(tmpdir):
    """Tests that the model can be trained on a pytorch geometric dataset."""
    coradataset = datasets.Planetoid(root="data", name="Cora")
    model = GraphNodeClassifier(num_features=coradataset.num_features, num_classes=coradataset.num_classes)
    datamodule = DataModule(
        GraphClassificationDatasetInput(
            RunningStage.TRAINING, coradataset, transform=GraphClassificationInputTransform
        ),
        batch_size=4,
    )
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model, datamodule=datamodule)


@pytest.mark.skipif(not _GRAPH_TESTING, reason="pytorch geometric isn't installed")
def test_val(tmpdir):
    """Tests that the model can be validated on a pytorch geometric dataset."""
    coradataset = datasets.Planetoid(root="data", name="Cora")
    model = GraphNodeClassifier(num_features=coradataset.num_features, num_classes=coradataset.num_classes)
    datamodule = DataModule(
        val_input=GraphClassificationDatasetInput(
            RunningStage.VALIDATING, coradataset, transform=GraphClassificationInputTransform
        ),
        batch_size=4,
    )
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.validate(model, datamodule=datamodule)


@pytest.mark.skipif(not _GRAPH_TESTING, reason="pytorch geometric isn't installed")
def test_test(tmpdir):
    """Tests that the model can be tested on a pytorch geometric dataset."""
    coradataset = datasets.Planetoid(root="data", name="Cora")
    model = GraphNodeClassifier(num_features=coradataset.num_features, num_classes=coradataset.num_classes)
    datamodule = DataModule(
        test_input=GraphClassificationDatasetInput(
            RunningStage.TESTING, coradataset, transform=GraphClassificationInputTransform
        ),
        batch_size=4,
    )
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.test(model, datamodule=datamodule)


@pytest.mark.skipif(not _GRAPH_TESTING, reason="pytorch geometric isn't installed")
def test_predict_dataset(tmpdir):
    """Tests that we can generate predictions from a pytorch geometric dataset."""
    coradataset = datasets.Planetoid(root="data", name="Cora")
    for data in coradataset:
        data.predict_mask = data.test_mask
    model = GraphNodeClassifier(num_features=coradataset.num_features, num_classes=coradataset.num_classes)
    datamodule = DataModule(
        predict_input=GraphClassificationDatasetInput(
            RunningStage.TESTING, coradataset, transform=GraphClassificationInputTransform
        ),
        batch_size=4,
    )
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    out = trainer.predict(model, datamodule=datamodule)
    assert isinstance(out[0][0], int)


@pytest.mark.skipif(not _GRAPH_TESTING, reason="pytorch geometric isn't installed")
def test_cli():
    cli_args = ["flash", "graph_node_classification", "--trainer.fast_dev_run", "True"]
    with mock.patch("sys.argv", cli_args):
        try:
            main()
        except SystemExit:
            pass
