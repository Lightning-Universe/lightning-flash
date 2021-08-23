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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, TYPE_CHECKING, Union

import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.utilities import rank_zero_warn

from flash.core.data.data_source import DefaultDataKeys, LabelsState
from flash.core.data.process import Serializer
from flash.core.model import Task
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE, lazy_import, requires

Classification, Classifications = None, None
if _FIFTYONE_AVAILABLE:
    fol = lazy_import("fiftyone.core.labels")
    if TYPE_CHECKING:
        from fiftyone.core.labels import Classification, Classifications
else:
    fol = None


def binary_cross_entropy_with_logits(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calls BCE with logits and cast the target one_hot (y) encoding to floating point precision."""
    return F.binary_cross_entropy_with_logits(x, y.float())


class ClassificationTask(Task):
    def __init__(
        self,
        *args,
        num_classes: Optional[int] = None,
        loss_fn: Optional[Callable] = None,
        metrics: Union[torchmetrics.Metric, Mapping, Sequence, None] = None,
        multi_label: bool = False,
        serializer: Optional[Union[Serializer, Mapping[str, Serializer]]] = None,
        **kwargs,
    ) -> None:
        if metrics is None:
            metrics = torchmetrics.F1(num_classes) if (multi_label and num_classes) else torchmetrics.Accuracy()

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


class PredsClassificationSerializer(ClassificationSerializer):
    """A :class:`~flash.core.classification.ClassificationSerializer` which gets the
    :attr:`~flash.core.data.data_source.DefaultDataKeys.PREDS` from the sample.
    """

    def serialize(self, sample: Any) -> Any:
        if isinstance(sample, Mapping) and DefaultDataKeys.PREDS in sample:
            sample = sample[DefaultDataKeys.PREDS]
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample)
        return sample


class Logits(PredsClassificationSerializer):
    """A :class:`.Serializer` which simply converts the model outputs (assumed to be logits) to a list."""

    def serialize(self, sample: Any) -> Any:
        return super().serialize(sample).tolist()


class Probabilities(PredsClassificationSerializer):
    """A :class:`.Serializer` which applies a softmax to the model outputs (assumed to be logits) and converts to a
    list."""

    def serialize(self, sample: Any) -> Any:
        sample = super().serialize(sample)
        if self.multi_label:
            return torch.sigmoid(sample).tolist()
        return torch.softmax(sample, -1).tolist()


class Classes(PredsClassificationSerializer):
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
        sample = super().serialize(sample)
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
        rank_zero_warn("No LabelsState was found, this serializer will act as a Classes serializer.", UserWarning)
        return classes


class FiftyOneLabels(ClassificationSerializer):
    """A :class:`.Serializer` which converts the model outputs to FiftyOne classification format.

    Args:
        labels: A list of labels, assumed to map the class index to the label for that class. If ``labels`` is not
            provided, will attempt to get them from the :class:`.LabelsState`.
        multi_label: If true, treats outputs as multi label logits.
        threshold: A threshold to use to filter candidate labels. In the single label case, predictions below this
            threshold will be replaced with None
        store_logits: Boolean determining whether to store logits in the FiftyOne labels
        return_filepath: Boolean determining whether to return a dict
            containing filepath and FiftyOne labels (True) or only a
            list of FiftyOne labels (False)
    """

    @requires("fiftyone")
    def __init__(
        self,
        labels: Optional[List[str]] = None,
        multi_label: bool = False,
        threshold: Optional[float] = None,
        store_logits: bool = False,
        return_filepath: bool = False,
    ):
        if multi_label and threshold is None:
            threshold = 0.5

        super().__init__(multi_label=multi_label)
        self._labels = labels
        self.threshold = threshold
        self.store_logits = store_logits
        self.return_filepath = return_filepath

        if labels is not None:
            self.set_state(LabelsState(labels))

    def serialize(
        self,
        sample: Any,
    ) -> Union[Classification, Classifications, Dict[str, Any], Dict[str, Any]]:
        pred = sample[DefaultDataKeys.PREDS] if isinstance(sample, Dict) else sample
        pred = torch.tensor(pred)

        labels = None

        if self._labels is not None:
            labels = self._labels
        else:
            state = self.get_state(LabelsState)
            if state is not None:
                labels = state.labels

        logits = None
        if self.store_logits:
            logits = pred.tolist()

        if self.multi_label:
            one_hot = (pred.sigmoid() > self.threshold).int().tolist()
            classes = []
            for index, value in enumerate(one_hot):
                if value == 1:
                    classes.append(index)
            probabilities = torch.sigmoid(pred).tolist()
        else:
            classes = torch.argmax(pred, -1).tolist()
            probabilities = torch.softmax(pred, -1).tolist()

        if labels is not None:
            if self.multi_label:
                classifications = []
                for idx in classes:
                    fo_cls = fol.Classification(
                        label=labels[idx],
                        confidence=probabilities[idx],
                    )
                    classifications.append(fo_cls)
                fo_predictions = fol.Classifications(
                    classifications=classifications,
                    logits=logits,
                )
            else:
                confidence = max(probabilities)
                if self.threshold is not None and confidence < self.threshold:
                    fo_predictions = None
                else:
                    fo_predictions = fol.Classification(
                        label=labels[classes],
                        confidence=confidence,
                        logits=logits,
                    )
        else:
            rank_zero_warn("No LabelsState was found, int targets will be used as label strings", UserWarning)

            if self.multi_label:
                classifications = []
                for idx in classes:
                    fo_cls = fol.Classification(
                        label=str(idx),
                        confidence=probabilities[idx],
                    )
                    classifications.append(fo_cls)
                fo_predictions = fol.Classifications(
                    classifications=classifications,
                    logits=logits,
                )
            else:
                confidence = max(probabilities)
                if self.threshold is not None and confidence < self.threshold:
                    fo_predictions = None
                else:
                    fo_predictions = fol.Classification(
                        label=str(classes),
                        confidence=confidence,
                        logits=logits,
                    )

        if self.return_filepath:
            filepath = sample[DefaultDataKeys.METADATA]["filepath"]
            return {"filepath": filepath, "predictions": fo_predictions}
        return fo_predictions
