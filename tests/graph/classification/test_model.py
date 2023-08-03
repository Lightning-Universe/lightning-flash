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
from flash import RunningStage, Trainer
from flash.core.data.data_module import DataModule
from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _TOPIC_GRAPH_AVAILABLE
from flash.graph.classification import GraphClassifier
from flash.graph.classification.input import GraphClassificationDatasetInput
from flash.graph.classification.input_transform import GraphClassificationInputTransform
from torch import Tensor

from tests.helpers.task_tester import TaskTester

if _TOPIC_GRAPH_AVAILABLE:
    from torch_geometric import datasets
    from torch_geometric.data import Batch, Data


class TestGraphClassifier(TaskTester):
    task = GraphClassifier
    task_kwargs = {"num_features": 1, "num_classes": 2}
    cli_command = "graph_classification"
    is_testing = _TOPIC_GRAPH_AVAILABLE
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
        assert output.shape == torch.Size([1, 2])

    @property
    def example_train_sample(self):
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
        return {DataKeys.INPUT: Data(x=x, edge_index=edge_index), DataKeys.TARGET: 1}

    @property
    def example_val_sample(self):
        return self.example_train_sample

    @property
    def example_test_sample(self):
        return self.example_train_sample


@pytest.mark.skipif(not _TOPIC_GRAPH_AVAILABLE, reason="pytorch geometric isn't installed")
def test_predict_dataset(tmpdir):
    """Tests that we can generate predictions from a pytorch geometric dataset."""
    tudataset = datasets.TUDataset(root=tmpdir, name="KKI")
    model = GraphClassifier(num_features=tudataset.num_features, num_classes=tudataset.num_classes)
    datamodule = DataModule(
        predict_input=GraphClassificationDatasetInput(RunningStage.TESTING, tudataset),
        transform=GraphClassificationInputTransform,
        batch_size=4,
    )
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    out = trainer.predict(model, datamodule=datamodule, output="classes")
    assert isinstance(out[0][0], int)
