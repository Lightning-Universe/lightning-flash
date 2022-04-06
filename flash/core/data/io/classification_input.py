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
from typing import Any, List, Optional

from flash.core.data.properties import Properties
from flash.core.data.utilities.classification import get_target_formatter, TargetFormatter


class ClassificationInputMixin(Properties):
    """The ``ClassificationInputMixin`` class provides utility methods for handling classification targets.
    :class:`~flash.core.data.io.input.Input` objects that extend ``ClassificationInputMixin`` should do the following:

    * In the ``load_data`` method, include a call to ``load_target_metadata``. This will determine the format of the
      targets and store metadata like ``labels`` and ``num_classes``.
    * In the ``load_sample`` method, use ``format_target`` to convert the target to a standard format for use with our
      tasks.
    """

    target_formatter: TargetFormatter
    multi_label: bool
    labels: list
    num_classes: int

    def load_target_metadata(
        self,
        targets: Optional[List[Any]],
        target_formatter: Optional[TargetFormatter] = None,
        add_background: bool = False,
    ) -> None:
        """Determine the target format and store the ``labels`` and ``num_classes``.

        Args:
            targets: The list of targets.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter`
                rather than inferring from the targets.
            add_background: If ``True``, a background class will be inserted as class zero if ``labels`` and
                ``num_classes`` are being inferred.
        """
        self.target_formatter = target_formatter
        if target_formatter is None and targets is not None:
            self.target_formatter = get_target_formatter(targets, add_background=add_background)

        if self.target_formatter is not None:
            self.multi_label = self.target_formatter.multi_label
            self.labels = self.target_formatter.labels
            self.num_classes = self.target_formatter.num_classes

    def format_target(self, target: Any) -> Any:
        """Format a single target according to the previously computed target format and metadata.

        Args:
            target: The target to format.

        Returns:
            The formatted target.
        """
        return getattr(self, "target_formatter", lambda x: x)(target)
