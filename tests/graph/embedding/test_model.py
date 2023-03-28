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
from typing import Any

import pytest
import torch
from torch import Tensor

from flash import RunningStage, Trainer
from flash.core.data.data_module import DataModule
from flash.core.utilities.imports import _GRAPH_TESTING, _TOPIC_GRAPH_AVAILABLE
from flash.graph.classification.input import GraphClassificationDatasetInput
from flash.graph.classification.input_transform import GraphClassificationInputTransform
from flash.graph.classification.model import GraphClassifier
from flash.graph.embedding.model import GraphEmbedder
from tests.helpers.task_tester import TaskTester

if _TOPIC_GRAPH_AVAILABLE:
    from torch_geometric import datasets
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn.models import GCN
else:
    GCN = dict


class TestGraphEmbedder(TaskTester):
    task = GraphEmbedder
    task_args = (GCN(in_channels=1, hidden_channels=512, num_layers=4),)
    is_testing = _GRAPH_TESTING
    is_available = _TOPIC_GRAPH_AVAILABLE

    # TODO: Resolve JIT issues
    scriptable = False
    traceable = False

    @property
    def example_forward_input(self):
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
        return Batch.from_data_list([Data(x=x, edge_index=edge_index)])

    def check_forward_output(self, output: Any):
        assert isinstance(output, Tensor)
        assert output.shape == torch.Size([1, 512])


@pytest.mark.skipif(not _GRAPH_TESTING, reason="pytorch geometric isn't installed")
def test_smoke():
    """A simple test that the class can be instantiated from a GraphClassifier backbone."""
    model = GraphEmbedder(GraphClassifier(num_features=1, num_classes=1).backbone)
    assert model is not None


@pytest.mark.skipif(not _GRAPH_TESTING, reason="pytorch geometric isn't installed")
def test_not_trainable(tmpdir):
    """Tests that the model gives an error when training, validating, or testing."""
    tudataset = datasets.TUDataset(root=tmpdir, name="KKI")
    model = GraphEmbedder(GraphClassifier(num_features=1, num_classes=1).backbone)
    datamodule = DataModule(
        GraphClassificationDatasetInput(RunningStage.TRAINING, tudataset),
        GraphClassificationDatasetInput(RunningStage.VALIDATING, tudataset),
        GraphClassificationDatasetInput(RunningStage.TESTING, tudataset),
        transform=GraphClassificationInputTransform,
        batch_size=4,
    )
    trainer = Trainer(default_root_dir=tmpdir, num_sanity_val_steps=0)
    with pytest.raises(NotImplementedError, match="Training a `GraphEmbedder` is not supported."):
        trainer.fit(model, datamodule=datamodule)

    with pytest.raises(NotImplementedError, match="Validating a `GraphEmbedder` is not supported."):
        trainer.validate(model, datamodule=datamodule)

    with pytest.raises(NotImplementedError, match="Testing a `GraphEmbedder` is not supported."):
        trainer.test(model, datamodule=datamodule)


@pytest.mark.skipif(not _GRAPH_TESTING, reason="pytorch geometric isn't installed")
def test_predict_dataset(tmpdir):
    """Tests that we can generate embeddings from a pytorch geometric dataset."""
    tudataset = datasets.TUDataset(root=tmpdir, name="KKI")
    model = GraphEmbedder(
        GraphClassifier(num_features=tudataset.num_features, num_classes=tudataset.num_classes).backbone
    )
    datamodule = DataModule(
        predict_input=GraphClassificationDatasetInput(RunningStage.PREDICTING, tudataset),
        transform=GraphClassificationInputTransform,
        batch_size=4,
    )
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    out = trainer.predict(model, datamodule=datamodule)
    assert isinstance(out[0][0], Tensor)
