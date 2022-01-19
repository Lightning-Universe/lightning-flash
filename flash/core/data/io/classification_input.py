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
from typing import Any, List, Optional, Sequence

from flash.core.data.properties import ProcessState, Properties
from flash.core.data.utilities.classification import get_target_formatter, TargetFormatter


@dataclass(unsafe_hash=True, frozen=True)
class ClassificationState(ProcessState):
    """A :class:`~flash.core.data.properties.ProcessState` containing ``labels`` (a mapping from class index to
    label) and ``num_classes``."""

    labels: Optional[Sequence[str]]
    num_classes: Optional[int] = None


class ClassificationInputMixin(Properties):
    """The ``ClassificationInputMixin`` class provides utility methods for handling classification targets.
    :class:`~flash.core.data.io.input.Input` objects that extend ``ClassificationInputMixin`` should do the following:

    * In the ``load_data`` method, include a call to ``load_target_metadata``. This will determine the format of the
      targets and store metadata like ``labels`` and ``num_classes``.
    * In the ``load_sample`` method, use ``format_target`` to convert the target to a standard format for use with our
      tasks.
    """

    def load_target_metadata(
        self, targets: Optional[List[Any]], target_formatter: Optional[TargetFormatter] = None
    ) -> None:
        """Determine the target format and store the ``labels`` and ``num_classes``.

        Args:
            targets: The list of targets.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter`
                rather than inferring from the targets.
        """
        if target_formatter is None and targets is not None:
            classification_state = self.get_state(ClassificationState)
            if classification_state is not None:
                labels, num_classes = classification_state.labels, classification_state.num_classes
            else:
                labels, num_classes = None, None

            self.target_formatter = get_target_formatter(targets, labels, num_classes)
        else:
            self.target_formatter = target_formatter

        if getattr(self, "target_formatter", None) is not None:
            self.multi_label = self.target_formatter.multi_label
            self.labels = self.target_formatter.labels
            self.num_classes = self.target_formatter.num_classes
            self.set_state(ClassificationState(self.labels, self.num_classes))

    def format_target(self, target: Any) -> Any:
        """Format a single target according to the previously computed target format and metadata.

        Args:
            target: The target to format.

        Returns:
            The formatted target.
        """
        if getattr(self, "target_formatter", None) is not None:
            return self.target_formatter(target)
        return target
