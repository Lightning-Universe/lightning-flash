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
from dataclasses import dataclass
from typing import Dict, Optional, Type, TypeVar

from pytorch_lightning.trainer.states import RunningStage

import flash


@dataclass(unsafe_hash=True, frozen=True)
class ProcessState:
    """Base class for all process states."""


STATE_TYPE = TypeVar("STATE_TYPE", bound=ProcessState)


class Properties:
    def __init__(self):
        super().__init__()

        self._running_stage: Optional[RunningStage] = None
        self._current_fn: Optional[str] = None
        self._data_pipeline_state: Optional["flash.core.data.data_pipeline.DataPipelineState"] = None
        self._state: Dict[Type[ProcessState], ProcessState] = {}

    def get_state(self, state_type: Type[STATE_TYPE]) -> Optional[STATE_TYPE]:
        if state_type in self._state:
            return self._state[state_type]
        if self._data_pipeline_state is not None:
            return self._data_pipeline_state.get_state(state_type)
        return None

    def set_state(self, state: ProcessState):
        self._state[type(state)] = state
        if self._data_pipeline_state is not None:
            self._data_pipeline_state.set_state(state)

    def attach_data_pipeline_state(self, data_pipeline_state: "flash.core.data.data_pipeline.DataPipelineState"):
        self._data_pipeline_state = data_pipeline_state
        for state in self._state.values():
            self._data_pipeline_state.set_state(state)

    @property
    def current_fn(self) -> Optional[str]:
        return self._current_fn

    @current_fn.setter
    def current_fn(self, current_fn: str):
        self._current_fn = current_fn

    @property
    def running_stage(self) -> Optional[RunningStage]:
        return self._running_stage

    @running_stage.setter
    def running_stage(self, running_stage: RunningStage):
        self._running_stage = running_stage

    @property
    def training(self) -> bool:
        return self._running_stage == RunningStage.TRAINING

    @training.setter
    def training(self, val: bool) -> None:
        if val:
            self._running_stage = RunningStage.TRAINING
        elif self.training:
            self._running_stage = None

    @property
    def testing(self) -> bool:
        return self._running_stage == RunningStage.TESTING

    @testing.setter
    def testing(self, val: bool) -> None:
        if val:
            self._running_stage = RunningStage.TESTING
        elif self.testing:
            self._running_stage = None

    @property
    def predicting(self) -> bool:
        return self._running_stage == RunningStage.PREDICTING

    @predicting.setter
    def predicting(self, val: bool) -> None:
        if val:
            self._running_stage = RunningStage.PREDICTING
        elif self.predicting:
            self._running_stage = None

    @property
    def validating(self) -> bool:
        return self._running_stage == RunningStage.VALIDATING

    @validating.setter
    def validating(self, val: bool) -> None:
        if val:
            self._running_stage = RunningStage.VALIDATING
        elif self.validating:
            self._running_stage = None
