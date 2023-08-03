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
from flash.core.data.properties import Properties
from flash.core.utilities.imports import _TOPIC_CORE_AVAILABLE
from flash.core.utilities.stages import RunningStage


@pytest.mark.skipif(not _TOPIC_CORE_AVAILABLE, reason="Not testing core.")
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
