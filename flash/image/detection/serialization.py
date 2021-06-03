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

class FiftyOneDetectionLabels(Serializer):
    """A :class:`.Serializer` which converts the model outputs to a FiftyOne Detections label.
    """

    def __init__(
            self,
            labels: Optional[List[str]] = None,
        ):
        if not _FIFTYONE_AVAILABLE:
            raise ModuleNotFoundError("Please, run `pip install fiftyone`.")
        super().__init__()
        self._labels = labels

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
                rank_zero_warn("No LabelsState was found, this serializer will return integer class labels.", UserWarning)

        detections = []

        for det in sample:
            xmin, ymin, xmax, ymax = [c.tolist() for c in det["boxes"]]
            box = [xmin, ymin, xmax-xmin, ymax-ymin]

            label = det["labels"].tolist()
            if labels is not None:
                label = labels[label]
            else:
                label = str(int(label))

            score = det["scores"].tolist()
            detections.append(Detection(
                label = label,
                bounding_box = box,
                confidence = score,
            ))


        return Detections(detections=detections)
