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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_warn
from torch import Tensor
from torchmetrics import Accuracy, Metric

from flash.core.adapter import AdapterTask
from flash.core.data.io.input import DataKeys
from flash.core.data.io.output import Output
from flash.core.model import Task
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE, _TM_GREATER_EQUAL_0_7_0, lazy_import, requires
from flash.core.utilities.providers import _FIFTYONE

if _FIFTYONE_AVAILABLE:
    fol = lazy_import("fiftyone.core.labels")
    Classification = "fiftyone.core.labels.Classification"
    Classifications = "fiftyone.core.labels.Classifications"
else:
    fol = None
    Classification = None
    Classifications = None

if _TM_GREATER_EQUAL_0_7_0:
    from torchmetrics import F1Score
else:
    from torchmetrics import F1 as F1Score


CLASSIFICATION_OUTPUTS = FlashRegistry("outputs")


def binary_cross_entropy_with_logits(x: Tensor, y: Tensor) -> Tensor:
    """Calls BCE with logits and cast the target one_hot (y) encoding to floating point precision."""
    return F.binary_cross_entropy_with_logits(x, y.float())


class ClassificationMixin:
    def _build(
        self,
        num_classes: Optional[int] = None,
        labels: Optional[List[str]] = None,
        loss_fn: Optional[Callable] = None,
        metrics: Union[Metric, Mapping, Sequence, None] = None,
        multi_label: bool = False,
    ):
        self.num_classes = num_classes
        self.multi_label = multi_label
        self.labels = labels

        if metrics is None:
            metrics = F1Score(num_classes) if (multi_label and num_classes) else Accuracy()

        if loss_fn is None:
            loss_fn = binary_cross_entropy_with_logits if multi_label else F.cross_entropy

        return metrics, loss_fn

    def to_metrics_format(self, x: Tensor) -> Tensor:
        if getattr(self, "multi_label", False):
            return torch.sigmoid(x)
        return torch.softmax(x, dim=1)


class ClassificationTask(ClassificationMixin, Task):

    outputs: FlashRegistry = Task.outputs + CLASSIFICATION_OUTPUTS

    def __init__(
        self,
        *args,
        num_classes: Optional[int] = None,
        loss_fn: Optional[Callable] = None,
        metrics: Union[Metric, Mapping, Sequence, None] = None,
        multi_label: bool = False,
        labels: Optional[List[str]] = None,
        **kwargs,
    ) -> None:

        metrics, loss_fn = self._build(num_classes, labels, loss_fn, metrics, multi_label)

        super().__init__(
            *args,
            loss_fn=loss_fn,
            metrics=metrics,
            **kwargs,
        )


class ClassificationAdapterTask(ClassificationMixin, AdapterTask):

    outputs: FlashRegistry = Task.outputs + CLASSIFICATION_OUTPUTS

    def __init__(
        self,
        *args,
        num_classes: Optional[int] = None,
        loss_fn: Optional[Callable] = None,
        metrics: Union[Metric, Mapping, Sequence, None] = None,
        multi_label: bool = False,
        labels: Optional[List[str]] = None,
        **kwargs,
    ) -> None:

        metrics, loss_fn = self._build(num_classes, labels, loss_fn, metrics, multi_label)

        super().__init__(
            *args,
            loss_fn=loss_fn,
            metrics=metrics,
            **kwargs,
        )


class ClassificationOutput(Output):
    """A base class for classification outputs.

    Args:
        multi_label: If true, treats outputs as multi label logits.
    """

    def __init__(self, multi_label: bool = False):
        super().__init__()

        self._mutli_label = multi_label

    @classmethod
    def from_task(cls, task: Task, **kwargs) -> Output:
        return cls(multi_label=getattr(task, "multi_label", False))

    @property
    def multi_label(self) -> bool:
        return self._mutli_label


@CLASSIFICATION_OUTPUTS(name="preds")
class PredsClassificationOutput(ClassificationOutput):
    """A :class:`~flash.core.classification.ClassificationOutput` which gets the
    :attr:`~flash.core.data.io.input.InputFormat.PREDS` from the sample.
    """

    def transform(self, sample: Any) -> Any:
        if isinstance(sample, Mapping) and DataKeys.PREDS in sample:
            sample = sample[DataKeys.PREDS]
        if not isinstance(sample, Tensor):
            sample = torch.tensor(sample)
        return sample


@CLASSIFICATION_OUTPUTS(name="logits")
class LogitsOutput(PredsClassificationOutput):
    """A :class:`.Output` which simply converts the model outputs (assumed to be logits) to a list."""

    def transform(self, sample: Any) -> Any:
        return super().transform(sample).tolist()


@CLASSIFICATION_OUTPUTS(name="probabilities")
class ProbabilitiesOutput(PredsClassificationOutput):
    """A :class:`.Output` which applies a softmax to the model outputs (assumed to be logits) and converts to a
    list."""

    def transform(self, sample: Any) -> Any:
        sample = super().transform(sample)
        if self.multi_label:
            return torch.sigmoid(sample).tolist()
        return torch.softmax(sample, -1).tolist()


@CLASSIFICATION_OUTPUTS(name="classes")
class ClassesOutput(PredsClassificationOutput):
    """A :class:`.Output` which applies an argmax to the model outputs (either logits or probabilities) and
    converts to a list.

    Args:
        multi_label: If true, treats outputs as multi label logits.
        threshold: The threshold to use for multi_label classification.
    """

    def __init__(self, multi_label: bool = False, threshold: float = 0.5):
        super().__init__(multi_label)

        self.threshold = threshold

    def transform(self, sample: Any) -> Union[int, List[int]]:
        sample = super().transform(sample)
        if self.multi_label:
            one_hot = (sample.sigmoid() > self.threshold).int().tolist()
            result = []
            for index, value in enumerate(one_hot):
                if value == 1:
                    result.append(index)
            return result
        return torch.argmax(sample, -1).tolist()


@CLASSIFICATION_OUTPUTS(name="labels")
class LabelsOutput(ClassesOutput):
    """A :class:`.Output` which converts the model outputs (either logits or probabilities) to the label of the
    argmax classification.

    Args:
        labels: A list of labels, assumed to map the class index to the label for that class.
        multi_label: If true, treats outputs as multi label logits.
        threshold: The threshold to use for multi_label classification.
    """

    def __init__(self, labels: Optional[List[str]] = None, multi_label: bool = False, threshold: float = 0.5):
        super().__init__(multi_label=multi_label, threshold=threshold)
        self._labels = labels

    @classmethod
    def from_task(cls, task: Task, **kwargs) -> Output:
        return cls(labels=getattr(task, "labels", None), multi_label=getattr(task, "multi_label", False))

    def transform(self, sample: Any) -> Union[int, List[int], str, List[str]]:
        classes = super().transform(sample)

        if self._labels is not None:
            if self.multi_label:
                return [self._labels[cls] for cls in classes]
            return self._labels[classes]
        rank_zero_warn("No labels were provided, this output will act as a Classes output.", category=UserWarning)
        return classes


@CLASSIFICATION_OUTPUTS(name="fiftyone", providers=_FIFTYONE)
class FiftyOneLabelsOutput(ClassificationOutput):
    """A :class:`.Output` which converts the model outputs to FiftyOne classification format.

    Args:
        labels: A list of labels, assumed to map the class index to the label for that class.
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
        return_filepath: bool = True,
    ):
        if multi_label and threshold is None:
            threshold = 0.5

        super().__init__(multi_label=multi_label)
        self._labels = labels
        self.threshold = threshold
        self.store_logits = store_logits
        self.return_filepath = return_filepath

    @classmethod
    def from_task(cls, task: Task, **kwargs) -> Output:
        return cls(labels=getattr(task, "labels", None), multi_label=getattr(task, "multi_label", False))

    def transform(
        self,
        sample: Any,
    ) -> Union[Classification, Classifications, Dict[str, Any]]:
        pred = sample[DataKeys.PREDS] if isinstance(sample, Dict) else sample
        pred = torch.tensor(pred)

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

        if self._labels is not None:
            if self.multi_label:
                classifications = []
                for idx in classes:
                    fo_cls = fol.Classification(
                        label=self._labels[idx],
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
                        label=self._labels[classes],
                        confidence=confidence,
                        logits=logits,
                    )
        else:
            rank_zero_warn("No labels were provided, int targets will be used as label strings.", category=UserWarning)

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
            filepath = sample[DataKeys.METADATA]["filepath"]
            return {"filepath": filepath, "predictions": fo_predictions}
        return fo_predictions
