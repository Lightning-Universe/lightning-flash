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
from typing import Any, Dict, List, Optional

from pytorch_lightning.utilities import rank_zero_warn

from flash.core.data.data_source import LabelsState
from flash.core.data.process import Serializer
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE

if _FIFTYONE_AVAILABLE:
    from fiftyone.core.labels import Detection, Detections
else:
    Detection, Detections = None, None


class FiftyOneDetectionLabels(Serializer):
    """A :class:`.Serializer` which converts model outputs to FiftyOne detection format.

    Args:
        labels: A list of labels, assumed to map the class index to the label for that class. If ``labels`` is not
            provided, will attempt to get them from the :class:`.LabelsState`.
        threshold: a score threshold to apply to candidate detections.
    """

    def __init__(self, labels: Optional[List[str]] = None, threshold: Optional[float] = None):
        if not _FIFTYONE_AVAILABLE:
            raise ModuleNotFoundError("Please, run `pip install fiftyone`.")

        super().__init__()
        self._labels = labels
        self.threshold = threshold

        if labels is not None:
            self.set_state(LabelsState(labels))

    def serialize(self, sample: List[Dict[str, Any]]) -> Detections:
        labels = None
        if self._labels is not None:
            labels = self._labels
        else:
            state = self.get_state(LabelsState)
            if state is not None:
                labels = state.labels
            else:
                rank_zero_warn("No LabelsState was found, int targets will be used as label strings", UserWarning)

        detections = []

        for det in sample:
            confidence = det["scores"].tolist()

            if self.threshold is not None and confidence < self.threshold:
                continue

            xmin, ymin, xmax, ymax = [c.tolist() for c in det["boxes"]]
            box = [xmin, ymin, xmax - xmin, ymax - ymin]

            label = det["labels"].tolist()
            if labels is not None:
                label = labels[label]
            else:
                label = str(int(label))

            detections.append(
                Detection(
                    label=label,
                    bounding_box=box,
                    confidence=confidence,
                )
            )

        return Detections(detections=detections)
