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
from pytorch_lightning.utilities.enums import LightningEnum


class RunningStage(LightningEnum):
    """Enum for the current running stage.

    This stage complements :class:`TrainerFn` by specifying the current running stage for each function.
    More than one running stage value can be set while a :class:`TrainerFn` is running:

        - ``TrainerFn.FITTING`` - ``RunningStage.{SANITY_CHECKING,TRAINING,VALIDATING}``
        - ``TrainerFn.VALIDATING`` - ``RunningStage.VALIDATING``
        - ``TrainerFn.TESTING`` - ``RunningStage.TESTING``
        - ``TrainerFn.PREDICTING`` - ``RunningStage.PREDICTING``
        - ``TrainerFn.SERVING`` - ``RunningStage.SERVING``
        - ``TrainerFn.TUNING`` - ``RunningStage.{TUNING,SANITY_CHECKING,TRAINING,VALIDATING}``
    """

    TRAINING = "train"
    SANITY_CHECKING = "sanity_check"
    VALIDATING = "validate"
    TESTING = "test"
    PREDICTING = "predict"
    SERVING = "serve"
    TUNING = "tune"
