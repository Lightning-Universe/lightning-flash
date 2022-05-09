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
from typing import Optional

from flash.core.utilities.stages import RunningStage


class Properties:
    def __init__(
        self,
        running_stage: Optional[RunningStage] = None,
    ):
        super().__init__()

        self._running_stage = running_stage

    @property
    def running_stage(self) -> Optional[RunningStage]:
        return self._running_stage

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
    def validating(self) -> bool:
        return self._running_stage == RunningStage.VALIDATING

    @validating.setter
    def validating(self, val: bool) -> None:
        if val:
            self._running_stage = RunningStage.VALIDATING
        elif self.validating:
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
    def serving(self) -> bool:
        return self._running_stage == RunningStage.SERVING

    @serving.setter
    def serving(self, val: bool) -> None:
        if val:
            self._running_stage = RunningStage.SERVING
        elif self.serving:
            self._running_stage = None
