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
import os
from unittest.mock import Mock

import torch
from torch.utils.data import DataLoader

from flash import RunningStage
from flash.core.classification import LabelsOutput
from flash.core.data.data_pipeline import DataPipeline, DataPipelineState
from flash.core.data.input_transform import InputTransform
from flash.core.data.io.classification_input import ClassificationState
from flash.core.data.io.output import Output
from flash.core.model import Task
from flash.core.trainer import Trainer


def test_output():
    """Tests basic ``Output`` methods."""
    my_output = Output()

    assert my_output.transform("test") == "test"

    my_output.transform = Mock()
    my_output("test")
    my_output.transform.assert_called_once()


def test_saving_with_output(tmpdir):
    checkpoint_file = os.path.join(tmpdir, "tmp.ckpt")

    class CustomModel(Task):
        def __init__(self):
            super().__init__(model=torch.nn.Linear(1, 1), loss_fn=torch.nn.MSELoss())

    output = LabelsOutput(["a", "b"])
    model = CustomModel()
    trainer = Trainer(fast_dev_run=True)
    data_pipeline = DataPipeline(input_transform=InputTransform(RunningStage.TRAINING), output=output)
    data_pipeline.initialize()
    model.data_pipeline = data_pipeline
    assert isinstance(model.input_transform, InputTransform)
    dummy_data = DataLoader(list(zip(torch.arange(10, dtype=torch.float), torch.arange(10, dtype=torch.float))))
    trainer.fit(model, train_dataloader=dummy_data)
    trainer.save_checkpoint(checkpoint_file)
    model = CustomModel.load_from_checkpoint(checkpoint_file)
    assert isinstance(model._data_pipeline_state, DataPipelineState)
    assert model._data_pipeline_state._state[ClassificationState] == ClassificationState(["a", "b"])
