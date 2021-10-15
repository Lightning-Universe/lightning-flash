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
from typing import Any

from flash.core.data.properties import Properties


class Output(Properties):
    def transform(sample: Any) -> Any:
        """Transform the given sample into the desired output format.

        Args:
            sample: The output from the :class:`.OutputTransform`.

        Returns:
            The transformed output.
        """
        return sample

    def __init__(self):
        """A :class:`.Output` encapsulates a single ``transform`` method which is used to convert the model output
        into the desired output format when predicting or serving."""
        super().__init__()
        self._is_enabled = True

    def enable(self):
        """Enable serialization."""
        self._is_enabled = True

    def disable(self):
        """Disable serialization."""
        self._is_enabled = False

    def __call__(self, sample: Any) -> Any:
        if self._is_enabled:
            return self.transform(sample)
        return sample
