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
from typing import Iterable, Optional, Union

from torch.nn import Module


class FineTuningHooks:
    """Hooks to be used in Task and FlashBaseTuning."""

    def modules_to_freeze(self) -> Optional[Union[Module, Iterable[Union[Module, Iterable]]]]:
        """Return the name(s) of the module attributes of the model to be frozen."""
        return None
