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

import pytest
import torch
from torch.utils.data import DataLoader

from flash.core.classification import Labels
from flash.core.data.data_pipeline import DataPipeline, DataPipelineState
from flash.core.data.data_source import LabelsState
from flash.core.data.io.output import Output, OutputMapping
from flash.core.data.process import DefaultPreprocess
from flash.core.data.properties import ProcessState
from flash.core.model import Task
from flash.core.trainer import Trainer


def test_output_enable_disable():
    """Tests that ``Output`` can be enabled and disabled correctly."""

    my_output = Output()

    assert my_output.transform("test") == "test"
    my_output.transform = Mock()

    my_output.disable()
    assert my_output("test") == "test"
    my_output.transform.assert_not_called()

    my_output.enable()
    my_output("test")
    my_output.transform.assert_called_once()


def test_saving_with_output(tmpdir):
    checkpoint_file = os.path.join(tmpdir, "tmp.ckpt")

    class CustomModel(Task):
        def __init__(self):
            super().__init__(model=torch.nn.Linear(1, 1), loss_fn=torch.nn.MSELoss())

    output = Labels(["a", "b"])
    model = CustomModel()
    trainer = Trainer(fast_dev_run=True)
    data_pipeline = DataPipeline(preprocess=DefaultPreprocess(), output=output)
    data_pipeline.initialize()
    model.data_pipeline = data_pipeline
    assert isinstance(model.preprocess, DefaultPreprocess)
    dummy_data = DataLoader(list(zip(torch.arange(10, dtype=torch.float), torch.arange(10, dtype=torch.float))))
    trainer.fit(model, train_dataloader=dummy_data)
    trainer.save_checkpoint(checkpoint_file)
    model = CustomModel.load_from_checkpoint(checkpoint_file)
    assert isinstance(model._data_pipeline_state, DataPipelineState)
    assert model._data_pipeline_state._state[LabelsState] == LabelsState(["a", "b"])


def test_output_mapping():
    """Tests that ``OutputMapping`` correctly passes its inputs to the underlying outputs.

    Also checks that state is retrieved / loaded correctly.
    """

    output1 = Output()
    output1.transform = Mock(return_value="test1")

    class output1State(ProcessState):
        pass

    output2 = Output()
    output2.transform = Mock(return_value="test2")

    class output2State(ProcessState):
        pass

    output_mapping = OutputMapping({"key1": output1, "key2": output2})
    assert output_mapping({"key1": "output1", "key2": "output2"}) == {"key1": "test1", "key2": "test2"}
    output1.transform.assert_called_once_with("output1")
    output2.transform.assert_called_once_with("output2")

    with pytest.raises(ValueError, match="output must be a mapping"):
        output_mapping("not a mapping")

    output1_state = output1State()
    output2_state = output2State()

    output1.set_state(output1_state)
    output2.set_state(output2_state)

    data_pipeline_state = DataPipelineState()
    output_mapping.attach_data_pipeline_state(data_pipeline_state)

    assert output1._data_pipeline_state is data_pipeline_state
    assert output2._data_pipeline_state is data_pipeline_state

    assert data_pipeline_state.get_state(output1State) is output1_state
    assert data_pipeline_state.get_state(output2State) is output2_state
