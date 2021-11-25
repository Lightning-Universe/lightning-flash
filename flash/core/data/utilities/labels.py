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
from enum import auto, Enum
from typing import Any, List, TypeVar, Union

T = TypeVar("T")


class LabelMode(Enum):
    """The ``LabelMode`` Enum describes the different supported label formats for targets in Flash."""

    MULTI_LIST = auto()
    MUTLI_COMMA_DELIMITED = auto()
    SINGLE = auto()

    def __add__(self, other: "LabelMode") -> "LabelMode":
        """The purpose of the addition here is to reduce the ``LabelMode`` over multiple targets. If one label mode
        is a comma delimite string, then their sum will be also. Otherwise, we expect that both label modes are
        consistent.

        Raises:
            ValueError: If the two  label modes could not be resolved to a single mode.
        """
        if self is LabelMode.SINGLE and other is LabelMode.SINGLE:
            return LabelMode.SINGLE
        elif self is LabelMode.MUTLI_COMMA_DELIMITED and other is LabelMode.MUTLI_COMMA_DELIMITED:
            return LabelMode.MUTLI_COMMA_DELIMITED
        elif self is LabelMode.MUTLI_COMMA_DELIMITED and other is LabelMode.SINGLE:
            return LabelMode.MUTLI_COMMA_DELIMITED
        elif self is LabelMode.SINGLE and other is LabelMode.MUTLI_COMMA_DELIMITED:
            return LabelMode.MUTLI_COMMA_DELIMITED
        elif self is LabelMode.MULTI_LIST and other is LabelMode.MULTI_LIST:
            return LabelMode.MULTI_LIST
        raise ValueError(
            "Found inconsistent label modes. All targets should be either: single values, lists of values, or "
            "comma-delimited strings."
        )

    @classmethod
    def from_target(cls, target: Any) -> "LabelMode":
        """Determine the ``LabelMode`` for a given target.

        Args:
            target: A target that is one of: a single target, a list of targets, a comma delimited string.
        """
        if isinstance(target, str):
            # TODO: This could be a dangerous assumption if people happen to have a label that contains a comma
            if "," in target:
                return LabelMode.MUTLI_COMMA_DELIMITED
            else:
                return LabelMode.SINGLE
        elif isinstance(target, List):
            return LabelMode.MULTI_LIST
        return LabelMode.SINGLE


def get_label_mode(targets: List[Any]) -> LabelMode:
    """Aggregate the ``LabelMode`` for a list of targets.

    Args:
        targets: The list of targets to get the label mode for.

    Returns:
        The total ``LabelMode`` of the list of targets.
    """
    return sum(LabelMode.from_target(target) for target in targets)


@dataclass
class _Token:
    """The ``_Token`` dataclass is used to override the hash of a value to be the hash of it's string
    representation.

    This allows for using ``set`` with objects such as ``torch.Tensor`` which can have inconsistent hash codes.
    """

    value: Any

    def __eq__(self, other: "_Token") -> bool:
        return str(self.value) == str(other.value)

    def __hash__(self) -> int:
        return hash(str(self.value))


class LabelDetails:
    def __init__(self, labels: List[Any], label_mode: LabelMode):
        self.labels = labels
        self.label_mode = label_mode

        self.label_to_idx = {label: idx for idx, label in enumerate(labels)}
        self.is_multilabel = label_mode is LabelMode.MULTI_LIST or label_mode is LabelMode.MUTLI_COMMA_DELIMITED
        self.num_classes = len(labels)

    def format_target(self, target: Any):
        if self.label_mode is LabelMode.MUTLI_COMMA_DELIMITED:
            return [self.label_to_idx[t] for t in target.split(",")]
        elif self.label_mode is LabelMode.MULTI_LIST:
            return [self.label_to_idx[t] for t in target]
        return self.label_to_idx[target]


def get_label_details(labels: Union[List[T], List[List[T]]]) -> LabelDetails:
    """Finds and sorts the unique labels in a list of single or multi label targets.

    Args:
        labels: A list of single or multi-label targets.

    Returns:
        (labels, is_multilabel): Tuple containing the sorted list of unique targets / labels and a boolean indicating
        whether or not the targets were multilabel.
    """
    label_mode = get_label_mode(labels)

    tokens = []
    if label_mode is LabelMode.MUTLI_COMMA_DELIMITED:
        for label in labels:
            tokens.extend(label.split(","))
    elif label_mode is LabelMode.MULTI_LIST:
        for label in labels:
            tokens.extend(label)
    else:
        tokens = labels
    tokens = map(_Token, tokens)

    unique_tokens = list(set(tokens))
    labels = list(map(lambda token: token.value, unique_tokens))
    labels.sort()
    return LabelDetails(labels, label_mode)
