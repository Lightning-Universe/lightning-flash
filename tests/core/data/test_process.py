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
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader

from flash import Task, Trainer
from flash.core.classification import Labels, LabelsState
from flash.core.data.data_module import DataModule
from flash.core.data.data_pipeline import DataPipeline, DataPipelineState, DefaultPreprocess
from flash.core.data.io.input import InputFormat
from flash.core.data.process import Serializer, SerializerMapping
from flash.core.data.properties import ProcessState


def test_serializer():
    """Tests that ``Serializer`` can be enabled and disabled correctly."""

    my_serializer = Serializer()

    assert my_serializer.serialize("test") == "test"
    my_serializer.serialize = Mock()

    my_serializer.disable()
    assert my_serializer("test") == "test"
    my_serializer.serialize.assert_not_called()

    my_serializer.enable()
    my_serializer("test")
    my_serializer.serialize.assert_called_once()


def test_serializer_mapping():
    """Tests that ``SerializerMapping`` correctly passes its inputs to the underlying serializers.

    Also checks that state is retrieved / loaded correctly.
    """

    serializer1 = Serializer()
    serializer1.serialize = Mock(return_value="test1")

    class Serializer1State(ProcessState):
        pass

    serializer2 = Serializer()
    serializer2.serialize = Mock(return_value="test2")

    class Serializer2State(ProcessState):
        pass

    serializer_mapping = SerializerMapping({"key1": serializer1, "key2": serializer2})
    assert serializer_mapping({"key1": "serializer1", "key2": "serializer2"}) == {"key1": "test1", "key2": "test2"}
    serializer1.serialize.assert_called_once_with("serializer1")
    serializer2.serialize.assert_called_once_with("serializer2")

    with pytest.raises(ValueError, match="output must be a mapping"):
        serializer_mapping("not a mapping")

    serializer1_state = Serializer1State()
    serializer2_state = Serializer2State()

    serializer1.set_state(serializer1_state)
    serializer2.set_state(serializer2_state)

    data_pipeline_state = DataPipelineState()
    serializer_mapping.attach_data_pipeline_state(data_pipeline_state)

    assert serializer1._data_pipeline_state is data_pipeline_state
    assert serializer2._data_pipeline_state is data_pipeline_state

    assert data_pipeline_state.get_state(Serializer1State) is serializer1_state
    assert data_pipeline_state.get_state(Serializer2State) is serializer2_state


def test_saving_with_serializers(tmpdir):
    checkpoint_file = os.path.join(tmpdir, "tmp.ckpt")

    class CustomModel(Task):
        def __init__(self):
            super().__init__(model=torch.nn.Linear(1, 1), loss_fn=torch.nn.MSELoss())

    serializer = Labels(["a", "b"])
    model = CustomModel()
    trainer = Trainer(fast_dev_run=True)
    data_pipeline = DataPipeline(preprocess=DefaultPreprocess(), serializer=serializer)
    data_pipeline.initialize()
    model.data_pipeline = data_pipeline
    assert isinstance(model.preprocess, DefaultPreprocess)
    dummy_data = DataLoader(list(zip(torch.arange(10, dtype=torch.float), torch.arange(10, dtype=torch.float))))
    trainer.fit(model, train_dataloader=dummy_data)
    trainer.save_checkpoint(checkpoint_file)
    model = CustomModel.load_from_checkpoint(checkpoint_file)
    assert isinstance(model._data_pipeline_state, DataPipelineState)
    assert model._data_pipeline_state._state[LabelsState] == LabelsState(["a", "b"])


class CustomPreprocess(DefaultPreprocess):
    def __init__(self):
        super().__init__(
            data_sources={
                "test": Mock(return_value="test"),
                InputFormat.TENSORS: Mock(return_value="tensors"),
            },
            default_data_source="test",
        )


def test_data_source_of_name():
    preprocess = CustomPreprocess()

    assert preprocess.data_source_of_name("test")() == "test"
    assert preprocess.data_source_of_name(InputFormat.TENSORS)() == "tensors"
    assert preprocess.data_source_of_name("tensors")() == "tensors"
    assert preprocess.data_source_of_name("default")() == "test"

    with pytest.raises(MisconfigurationException, match="available data sources are: test, tensor"):
        preprocess.data_source_of_name("not available")


def test_available_data_sources():
    preprocess = CustomPreprocess()

    assert InputFormat.TENSORS in preprocess.available_data_sources()
    assert "test" in preprocess.available_data_sources()
    assert len(preprocess.available_data_sources()) == 3

    data_module = DataModule(preprocess=preprocess)

    assert InputFormat.TENSORS in data_module.available_data_sources()
    assert "test" in data_module.available_data_sources()
    assert len(data_module.available_data_sources()) == 3


def test_check_transforms():
    transform = torch.nn.Identity()
    DefaultPreprocess(train_transform=transform)
    DefaultPreprocess(train_transform=[transform])
