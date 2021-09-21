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
import flash
from flash.core.utilities.imports import requires
from flash.image.classification.integrations.baal.loop import ActiveLearningDataModule, ActiveLearningLoop  # noqa F401


class ActiveLearningTrainer(flash.Trainer):
    @requires("baal")
    def __init__(self, *args, label_epoch_frequency: int = 1, inference_iteration: int = 2, **kwags):
        super().__init__(*args, **kwags)

        # `connect` the `ActiveLearningLoop` to the Trainer
        active_learning_loop = ActiveLearningLoop(
            label_epoch_frequency=label_epoch_frequency, inference_iteration=inference_iteration
        )
        active_learning_loop.connect(self.fit_loop)
        self.fit_loop = active_learning_loop
        active_learning_loop.trainer = self
