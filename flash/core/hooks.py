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
from typing import List, Optional, Tuple, Union


class FineTuningHook:
    def get_parameters_to_freeze_before_training(self) -> Union[str, List[str]]:
        """Return the name(s) of the module attributes of the model to be frozen."""
        pass

    def determine_finetuning_strategy(self, strategy: str) -> Optional[Union[int, Tuple[Tuple[int, ...], int]]]:
        """Return the strategy along with the required parameters.

        One time definition for all types of strategies if required.
        """
        # Return value based on strategy
        #     No Freeze          = None
        #     Freeze             = None
        #     FreezeUnfreeze     = 10
        #     UnfreezeMilestones = ((5, 10), 5)
        return None
