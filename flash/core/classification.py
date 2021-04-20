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
from typing import Any, List, Mapping, Optional, Union

import torch
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_warn

from flash.core.model import Task
from flash.data.process import ProcessState, Serializer


@dataclass(unsafe_hash=True, frozen=True)
class ClassificationState(ProcessState):

    labels: Optional[List[str]]


class ClassificationTask(Task):

    def __init__(
        self,
        *args,
        serializer: Optional[Union[Serializer, Mapping[str, Serializer]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, serializer=serializer or Classes(), **kwargs)

    def to_metrics_format(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self.hparams, "multi_label", False):
            return F.sigmoid(x)
        return F.softmax(x, -1)


class ClassificationSerializer(Serializer):
    """A base class for classification serializers.

    Args:
        multi_label: If true, treats outputs as multi label logits.
    """

    def __init__(self, multi_label: bool = False):
        super().__init__()

        self._mutli_label = multi_label

    @property
    def multi_label(self) -> bool:
        return self._mutli_label


class Logits(ClassificationSerializer):
    """A :class:`.Serializer` which simply converts the model outputs (assumed to be logits) to a list."""

    def serialize(self, sample: Any) -> Any:
        return sample.tolist()


class Probabilities(ClassificationSerializer):
    """A :class:`.Serializer` which applies a softmax to the model outputs (assumed to be logits) and converts to a
    list."""

    def serialize(self, sample: Any) -> Any:
        if self.multi_label:
            return torch.sigmoid(sample).tolist()
        return torch.softmax(sample, -1).tolist()


class Classes(ClassificationSerializer):
    """A :class:`.Serializer` which applies an argmax to the model outputs (either logits or probabilities) and
    converts to a list."""

    def __init__(self, multi_label: bool = False, threshold: float = 0.0):
        super().__init__(multi_label)

        self.threshold = threshold

    def serialize(self, sample: Any) -> int:
        if self.multi_label:
            return (sample > self.threshold).int().tolist()
        return torch.argmax(sample, -1).tolist()


class Labels(Classes):
    """A :class:`.Serializer` which converts the model outputs (either logits or probabilities) to the label of the
    argmax classification.

    Args:
        labels: A list of labels, assumed to map the class index to the label for that class. If ``labels`` is not
            provided, will attempt to get them from the :class:`.ClassificationState`.
    """

    def __init__(self, labels: Optional[List[str]] = None):
        super().__init__(multi_label=False)  # TODO: Add support for multi-label
        self._labels = labels

    def serialize(self, sample: Any) -> Union[int, str]:
        labels = None

        if self._labels is not None:
            labels = self._labels
        else:
            state = self.get_state(ClassificationState)
            if state is not None:
                labels = state.labels

        cls = super().serialize(sample)

        if labels is not None:
            return labels[cls]
        else:
            rank_zero_warn(
                "No ClassificationState was found, this serializer will act as a Classes serializer.", UserWarning
            )
            return cls
