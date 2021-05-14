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
from typing import Any, Callable, List, Mapping, Optional, Sequence, Union

import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.utilities import rank_zero_warn

from flash.core.data.data_source import LabelsState
from flash.core.data.process import Serializer
from flash.core.model import Task


def binary_cross_entropy_with_logits(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calls BCE with logits and cast the target one_hot (y) encoding to floating point precision."""
    return F.binary_cross_entropy_with_logits(x, y.float())


class ClassificationTask(Task):

    def __init__(
        self,
        *args,
        loss_fn: Optional[Callable] = None,
        metrics: Union[torchmetrics.Metric, Mapping, Sequence, None] = None,
        multi_label: bool = False,
        serializer: Optional[Union[Serializer, Mapping[str, Serializer]]] = None,
        **kwargs,
    ) -> None:
        if metrics is None:
            metrics = torchmetrics.Accuracy(subset_accuracy=multi_label)

        if loss_fn is None:
            loss_fn = binary_cross_entropy_with_logits if multi_label else F.cross_entropy
        super().__init__(
            *args,
            loss_fn=loss_fn,
            metrics=metrics,
            serializer=serializer or Classes(multi_label=multi_label),
            **kwargs,
        )

    def to_metrics_format(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self.hparams, "multi_label", False):
            return torch.sigmoid(x)
        # we'll assume that the data always comes as `(B, C, ...)`
        return torch.softmax(x, dim=1)


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
    converts to a list.

    Args:
        multi_label: If true, treats outputs as multi label logits.

        threshold: The threshold to use for multi_label classification.
    """

    def __init__(self, multi_label: bool = False, threshold: float = 0.5):
        super().__init__(multi_label)

        self.threshold = threshold

    def serialize(self, sample: Any) -> Union[int, List[int]]:
        if self.multi_label:
            one_hot = (sample.sigmoid() > self.threshold).int().tolist()
            result = []
            for index, value in enumerate(one_hot):
                if value == 1:
                    result.append(index)
            return result
        return torch.argmax(sample, -1).tolist()


class Labels(Classes):
    """A :class:`.Serializer` which converts the model outputs (either logits or probabilities) to the label of the
    argmax classification.

    Args:
        labels: A list of labels, assumed to map the class index to the label for that class. If ``labels`` is not
            provided, will attempt to get them from the :class:`.LabelsState`.

        multi_label: If true, treats outputs as multi label logits.

        threshold: The threshold to use for multi_label classification.
    """

    def __init__(self, labels: Optional[List[str]] = None, multi_label: bool = False, threshold: float = 0.5):
        super().__init__(multi_label=multi_label, threshold=threshold)
        self._labels = labels

        if labels is not None:
            self.set_state(LabelsState(labels))

    def serialize(self, sample: Any) -> Union[int, List[int], str, List[str]]:
        labels = None

        if self._labels is not None:
            labels = self._labels
        else:
            state = self.get_state(LabelsState)
            if state is not None:
                labels = state.labels

        classes = super().serialize(sample)

        if labels is not None:
            if self.multi_label:
                return [labels[cls] for cls in classes]
            return labels[classes]
        else:
            rank_zero_warn("No LabelsState was found, this serializer will act as a Classes serializer.", UserWarning)
            return classes
