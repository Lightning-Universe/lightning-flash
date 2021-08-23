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
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

from pytorch_lightning.utilities import rank_zero_warn

from flash.core.data.data_source import DefaultDataKeys, LabelsState
from flash.core.data.process import Serializer
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE, lazy_import, requires

Detections = None
if _FIFTYONE_AVAILABLE:
    fo = lazy_import("fiftyone")
    if TYPE_CHECKING:
        from fiftyone import Detections
else:
    fo = None


class DetectionLabels(Serializer):
    """A :class:`.Serializer` which extracts predictions from sample dict."""

    def serialize(self, sample: Any) -> Dict[str, Any]:
        sample = sample[DefaultDataKeys.PREDS] if isinstance(sample, Dict) else sample
        return sample


class FiftyOneDetectionLabels(Serializer):
    """A :class:`.Serializer` which converts model outputs to FiftyOne detection format.

    Args:
        labels: A list of labels, assumed to map the class index to the label for that class. If ``labels`` is not
            provided, will attempt to get them from the :class:`.LabelsState`.
        threshold: a score threshold to apply to candidate detections.
        return_filepath: Boolean determining whether to return a dict
            containing filepath and FiftyOne labels (True) or only a
            list of FiftyOne labels (False)
    """

    @requires("fiftyone")
    def __init__(
        self,
        labels: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        return_filepath: bool = False,
    ):
        super().__init__()
        self._labels = labels
        self.threshold = threshold
        self.return_filepath = return_filepath

        if labels is not None:
            self.set_state(LabelsState(labels))

    def serialize(self, sample: Dict[str, Any]) -> Union[Detections, Dict[str, Any]]:
        if DefaultDataKeys.METADATA not in sample:
            raise ValueError("sample requires DefaultDataKeys.METADATA to use a FiftyOneDetectionLabels serializer.")

        labels = None
        if self._labels is not None:
            labels = self._labels
        else:
            state = self.get_state(LabelsState)
            if state is not None:
                labels = state.labels
            else:
                rank_zero_warn("No LabelsState was found, int targets will be used as label strings", UserWarning)

        height, width = sample[DefaultDataKeys.METADATA]["size"]

        detections = []

        for det in sample[DefaultDataKeys.PREDS]:
            confidence = det["scores"].tolist()

            if self.threshold is not None and confidence < self.threshold:
                continue

            xmin, ymin, xmax, ymax = (c.tolist() for c in det["boxes"])
            box = [
                xmin / width,
                ymin / height,
                (xmax - xmin) / width,
                (ymax - ymin) / height,
            ]

            label = det["labels"].tolist()
            if labels is not None:
                label = labels[label]
            else:
                label = str(int(label))

            detections.append(
                fo.Detection(
                    label=label,
                    bounding_box=box,
                    confidence=confidence,
                )
            )
        fo_predictions = fo.Detections(detections=detections)
        if self.return_filepath:
            filepath = sample[DefaultDataKeys.METADATA]["filepath"]
            return {"filepath": filepath, "predictions": fo_predictions}
        return fo_predictions
