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
import pytest

from flash.core.data.data_pipeline import DataPipelineState
from flash.core.data.properties import ProcessState, Properties
from flash.core.utilities.stages import RunningStage


def test_properties_data_pipeline_state():
    """Tests that ``get_state`` and ``set_state`` work for properties and that ``DataPipelineState`` is attached
    correctly."""

    class MyProcessState1(ProcessState):
        pass

    class MyProcessState2(ProcessState):
        pass

    class OtherProcessState(ProcessState):
        pass

    my_properties = Properties()
    my_properties.set_state(MyProcessState1())
    assert my_properties._state == {MyProcessState1: MyProcessState1()}
    assert my_properties.get_state(OtherProcessState) is None

    data_pipeline_state = DataPipelineState()
    data_pipeline_state.set_state(OtherProcessState())
    my_properties.attach_data_pipeline_state(data_pipeline_state)
    assert my_properties.get_state(OtherProcessState) == OtherProcessState()

    my_properties.set_state(MyProcessState2())
    assert data_pipeline_state.get_state(MyProcessState2) == MyProcessState2()


@pytest.mark.parametrize(
    "running_stage", [RunningStage.TRAINING, RunningStage.VALIDATING, RunningStage.TESTING, RunningStage.PREDICTING]
)
def test_properties_running_stage(running_stage):
    property_names = {
        RunningStage.TRAINING: "training",
        RunningStage.VALIDATING: "validating",
        RunningStage.TESTING: "testing",
        RunningStage.PREDICTING: "predicting",
    }

    my_properties = Properties()

    setattr(my_properties, property_names[running_stage], True)
    assert my_properties._running_stage == running_stage
    for stage in {RunningStage.TRAINING, RunningStage.VALIDATING, RunningStage.TESTING, RunningStage.PREDICTING}:
        if stage == running_stage:
            assert getattr(my_properties, property_names[stage])
        else:
            assert not getattr(my_properties, property_names[stage])
    setattr(my_properties, property_names[running_stage], False)
    assert my_properties._running_stage is None
