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
from functools import lru_cache
from typing import Any, List, Optional, Sequence

from flash.core.data.io.input_base import Input
from flash.core.data.properties import ProcessState
from flash.core.data.utilities.classification import (
    get_target_details,
    get_target_formatter,
    get_target_mode,
    TargetFormatter,
)


@dataclass(unsafe_hash=True, frozen=True)
class ClassificationState(ProcessState):
    """A :class:`~flash.core.data.properties.ProcessState` containing ``labels`` (a mapping from class index to
    label) and ``num_classes``."""

    labels: Optional[Sequence[str]]
    num_classes: Optional[int] = None


class ClassificationInput(Input):
    """The ``ClassificationInput`` class provides utility methods for handling classification targets.
    :class:`~flash.core.data.io.input_base.Input` objects that extend ``ClassificationInput`` should do the following:

    * In the ``load_data`` method, include a call to ``load_target_metadata``. This will determine the format of the
      targets and store metadata like ``labels`` and ``num_classes``.
    * In the ``load_sample`` method, use ``format_target`` to convert the target to a standard format for use with our
      tasks.
    """

    @property
    @lru_cache(maxsize=None)
    def target_formatter(self) -> TargetFormatter:
        """Get the :class:`~flash.core.data.utiltiies.classification.TargetFormatter` to use when formatting
        targets.

        This property uses ``functools.lru_cache`` so that we only instantiate the formatter once.
        """
        classification_state = self.get_state(ClassificationState)
        return get_target_formatter(self.target_mode, classification_state.labels, classification_state.num_classes)

    def load_target_metadata(self, targets: List[Any]) -> None:
        """Determine the target format and store the ``labels`` and ``num_classes``.

        Args:
            targets: The list of targets.
        """
        self.target_mode = get_target_mode(targets)
        if self.training:
            self.labels, self.num_classes = get_target_details(targets, self.target_mode)
            self.set_state(ClassificationState(self.labels, self.num_classes))

    def format_target(self, target: Any) -> Any:
        """Format a single target according to the previously computed target format and metadata.

        Args:
            target: The target to format.

        Returns:
            The formatted target.
        """
        return self.target_formatter(target)
