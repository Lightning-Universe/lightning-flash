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
import re
from enum import auto, Enum
from functools import reduce
from typing import Any, cast, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch


def _is_list_like(x: Any) -> bool:
    try:
        _ = x[0]
        _ = len(x)
        return True
    except (TypeError, IndexError):  # Single element tensors raise an `IndexError`
        return False


def _as_list(x: Union[List, torch.Tensor, np.ndarray]) -> List:
    if torch.is_tensor(x) or isinstance(x, np.ndarray):
        return cast(List, x.tolist())
    return x


def _convert(text: str) -> Union[int, str]:
    return int(text) if text.isdigit() else text


def _alphanumeric_key(key: str) -> List[Union[int, str]]:
    return [_convert(c) for c in re.split("([0-9]+)", key)]


def _sorted_nicely(iterable: Iterable[str]) -> Iterable[str]:
    """Sort the given iterable in the way that humans expect.

    Copied from:
    https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/
    """
    return sorted(iterable, key=_alphanumeric_key)


class TargetMode(Enum):
    """The ``TargetMode`` Enum describes the different supported formats for targets in Flash."""

    MULTI_TOKEN = auto()
    MULTI_NUMERIC = auto()
    MUTLI_COMMA_DELIMITED = auto()
    MULTI_BINARY = auto()

    SINGLE_TOKEN = auto()
    SINGLE_NUMERIC = auto()
    SINGLE_BINARY = auto()

    @classmethod
    def from_target(cls, target: Any) -> "TargetMode":
        """Determine the ``TargetMode`` for a given target.

        Multi-label targets can be:
            * Comma delimited string - ``TargetMode.MUTLI_COMMA_DELIMITED`` (e.g. ["blue,green", "red"])
            * List of strings - ``TargetMode.MULTI_TOKEN`` (e.g. [["blue", "green"], ["red"]])
            * List of numbers - ``TargetMode.MULTI_NUMERIC`` (e.g. [[0, 1], [2]])
            * Binary list - ``TargetMode.MULTI_BINARY`` (e.g. [[1, 1, 0], [0, 0, 1]])

        Single-label targets can be:
            * Single string - ``TargetMode.SINGLE_TOKEN`` (e.g. ["blue", "green", "red"])
            * Single number - ``TargetMode.SINGLE_NUMERIC`` (e.g. [0, 1, 2])
            * One-hot binary list - ``TargetMode.SINGLE_BINARY`` (e.g. [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        Args:
            target: A target that is one of: a single target, a list of targets, a comma delimited string.
        """
        if isinstance(target, str):
            # TODO: This could be a dangerous assumption if people happen to have a label that contains a comma
            if "," in target:
                return TargetMode.MUTLI_COMMA_DELIMITED
            else:
                return TargetMode.SINGLE_TOKEN
        elif _is_list_like(target):
            if isinstance(target[0], str):
                return TargetMode.MULTI_TOKEN
            elif all(t == 0 or t == 1 for t in target):
                if sum(target) == 1:
                    return TargetMode.SINGLE_BINARY
                return TargetMode.MULTI_BINARY
            return TargetMode.MULTI_NUMERIC
        return TargetMode.SINGLE_NUMERIC

    @property
    def is_multi_label(self) -> bool:
        return any(
            [
                self is TargetMode.MUTLI_COMMA_DELIMITED,
                self is TargetMode.MULTI_NUMERIC,
                self is TargetMode.MULTI_TOKEN,
                self is TargetMode.MULTI_BINARY,
            ]
        )

    @property
    def is_numeric(self) -> bool:
        return any(
            [
                self is TargetMode.MULTI_NUMERIC,
                self is TargetMode.SINGLE_NUMERIC,
            ]
        )

    @property
    def is_binary(self) -> bool:
        return any(
            [
                self is TargetMode.MULTI_BINARY,
                self is TargetMode.SINGLE_BINARY,
            ]
        )


_RESOLUTION_MAPPING = {
    TargetMode.MULTI_BINARY: [TargetMode.MULTI_NUMERIC],
    TargetMode.SINGLE_BINARY: [TargetMode.MULTI_BINARY, TargetMode.MULTI_NUMERIC],
    TargetMode.SINGLE_TOKEN: [TargetMode.MUTLI_COMMA_DELIMITED],
}


def _resolve_target_mode(a: TargetMode, b: TargetMode) -> TargetMode:
    """The purpose of the addition here is to reduce the ``TargetMode`` over multiple targets. If one target mode
    is a comma delimited string and the other a single string then their sum will be comma delimited. If one target
    is multi binary and the other is single binary, their sum will be multi binary. Otherwise, we expect that both
    target modes are the same.

    Raises:
        ValueError: If the two  target modes could not be resolved to a single mode.
    """
    if a is b:
        return a
    elif a in _RESOLUTION_MAPPING and b in _RESOLUTION_MAPPING[a]:
        return b
    elif b in _RESOLUTION_MAPPING and a in _RESOLUTION_MAPPING[b]:
        return a
    raise ValueError(
        "Found inconsistent target modes. All targets should be either: single values, lists of values, or "
        "comma-delimited strings."
    )


def get_target_mode(targets: List[Any]) -> TargetMode:
    """Aggregate the ``TargetMode`` for a list of targets.

    Args:
        targets: The list of targets to get the label mode for.

    Returns:
        The total ``TargetMode`` of the list of targets.
    """
    targets = _as_list(targets)
    return reduce(_resolve_target_mode, [TargetMode.from_target(target) for target in targets])


class TargetFormatter:
    def __call__(self, target: Any) -> Any:
        return self.format(target)

    def format(self, target: Any) -> Any:
        return _as_list(target)


class SingleLabelFormatter(TargetFormatter):
    def __init__(self, labels: List[Any]):
        self.label_to_idx = {label: idx for idx, label in enumerate(labels)}

    def format(self, target: Any) -> Any:
        return self.label_to_idx[(target[0] if not isinstance(target, str) else target).strip()]


class MultiLabelFormatter(SingleLabelFormatter):
    def __init__(self, labels: List[Any]):
        super().__init__(labels)

        self.num_classes = len(labels)

    def format(self, target: Any) -> Any:
        result = [0] * self.num_classes
        for t in target:
            idx = super().format(t)
            result[idx] = 1
        return result


class CommaDelimitedFormatter(MultiLabelFormatter):
    def format(self, target: Any) -> Any:
        return super().format(target.split(","))


class MultiNumericFormatter(TargetFormatter):
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def format(self, target: Any) -> Any:
        result = [0] * self.num_classes
        for idx in target:
            result[idx] = 1
        return result


class OneHotFormatter(TargetFormatter):
    def format(self, target: Any) -> Any:
        for idx, t in enumerate(target):
            if t == 1:
                return idx
        return 0


def get_target_formatter(
    target_mode: TargetMode, labels: Optional[List[Any]], num_classes: Optional[int]
) -> TargetFormatter:
    if target_mode is TargetMode.SINGLE_NUMERIC or target_mode is TargetMode.MULTI_BINARY:
        return TargetFormatter()
    elif target_mode is TargetMode.SINGLE_BINARY:
        return OneHotFormatter()
    elif target_mode is TargetMode.MULTI_NUMERIC:
        return MultiNumericFormatter(num_classes)
    elif target_mode is TargetMode.SINGLE_TOKEN:
        return SingleLabelFormatter(labels)
    elif target_mode is TargetMode.MUTLI_COMMA_DELIMITED:
        return CommaDelimitedFormatter(labels)
    return MultiLabelFormatter(labels)


def get_target_details(targets: List[Any], target_mode: TargetMode) -> Tuple[Optional[List[Any]], Optional[int]]:
    """Finds and sorts the unique labels in a list of single or multi label targets.

    Args:
        targets: A list of single or multi-label targets.

    Returns:
        (labels, is_multilabel): Tuple containing the sorted list of unique targets / labels and a boolean indicating
        whether or not the targets were multilabel.
    """

    # Multi-label targets can be:
    # Comma delimited string (e.g. ["blue,green", "red"]) -> Count unique tokens
    # List of strings (e.g. [["blue", "green"], ["red"]]) -> Count unique tokens
    # List of numbers (e.g. [[0, 1], [2]]) -> Take a max
    # Binary list (e.g. [[1, 1, 0], [0, 0, 1]]) -> Take the length

    # Single-label targets can be:
    # Single string (e.g. ["blue", "green", "red"]) -> Count unique tokens
    # Single number (e.g. [0, 1, 2]) -> Take a max
    # One-hot binary list (e.g. [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) -> Take the length

    if target_mode.is_numeric:
        # Take a max over all values
        if target_mode is TargetMode.MULTI_NUMERIC:
            values = []
            for target in targets:
                values.extend(target)
        else:
            values = targets
        num_classes = max(values)
        if _is_list_like(num_classes):
            num_classes = num_classes[0]
        num_classes = num_classes + 1
        labels = None
    elif target_mode.is_binary:
        # Take a length
        # TODO: Add a check here and error if target lengths are not all equal
        num_classes = len(targets[0])
        labels = None
    else:
        # Compute tokens
        tokens = []
        if target_mode is TargetMode.MUTLI_COMMA_DELIMITED:
            for target in targets:
                tokens.extend(target.split(","))
        elif target_mode is TargetMode.MULTI_TOKEN:
            for target in targets:
                tokens.extend(target)
        else:
            tokens = targets

        tokens = [token.strip() for token in tokens]
        labels = list(_sorted_nicely(set(tokens)))
        num_classes = len(labels)
    return labels, num_classes
