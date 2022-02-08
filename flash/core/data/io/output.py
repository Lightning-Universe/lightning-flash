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
from abc import abstractmethod
from typing import Any

import flash
from flash.core.data.properties import Properties


class Output(Properties):
    """An :class:`.Output` encapsulates a single :meth:`~flash.core.data.io.output.Output.transform` method which
    is used to convert the model output into the desired output format when predicting."""

    @classmethod
    @abstractmethod
    def from_task(cls, task: "flash.Task", **kwargs) -> "Output":
        """Instantiate the output from the given :class:`~flash.core.model.Task`."""
        return cls()

    @staticmethod
    def transform(sample: Any) -> Any:
        """Convert the given sample into the desired output format.

        Args:
            sample: The output from the :class:`.OutputTransform`.

        Returns:
            The converted output.
        """
        return sample

    def __call__(self, sample: Any) -> Any:
        return self.transform(sample)
