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
from unittest.mock import Mock

import pytest

from flash.data.data_pipeline import DataPipelineState
from flash.data.process import ProcessState, Serializer, SerializerMapping


def test_properties_data_pipeline_state():
    pass


def test_serializer():
    my_serializer = Serializer()

    assert my_serializer.serialize('test') == 'test'
    my_serializer.serialize = Mock()

    my_serializer.disable()
    assert my_serializer('test') == 'test'
    my_serializer.serialize.assert_not_called()

    my_serializer.enable()
    my_serializer('test')
    my_serializer.serialize.assert_called_once()


def test_serializer_mapping():
    serializer1 = Serializer()
    serializer1.serialize = Mock(return_value='test1')

    class Serializer1State(ProcessState):
        pass

    serializer2 = Serializer()
    serializer2.serialize = Mock(return_value='test2')

    class Serializer2State(ProcessState):
        pass

    serializer_mapping = SerializerMapping({'key1': serializer1, 'key2': serializer2})
    assert serializer_mapping({'key1': 'serializer1', 'key2': 'serializer2'}) == {'key1': 'test1', 'key2': 'test2'}
    serializer1.serialize.assert_called_once_with('serializer1')
    serializer2.serialize.assert_called_once_with('serializer2')

    with pytest.raises(ValueError, match='output must be a mapping'):
        serializer_mapping('not a mapping')

    test_state = {
        'key1': {
            Serializer1State: Serializer1State()
        },
        'key2': {
            Serializer2State: Serializer2State()
        },
    }

    serializer_mapping.state = test_state
    assert serializer1.state == test_state['key1']
    assert serializer2.state == test_state['key2']

    assert serializer_mapping.state == test_state

    data_pipeline_state = DataPipelineState()
    serializer_mapping.attach_data_pipeline_state(data_pipeline_state)

    assert serializer1._data_pipeline_state is data_pipeline_state
    assert serializer2._data_pipeline_state is data_pipeline_state
