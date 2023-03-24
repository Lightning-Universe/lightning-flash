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
from typing import Any, Dict, List, Optional, Union

from flash.core.data.io.input import DataKeys
from flash.core.data.io.output import Output
from flash.core.model import Task
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE, lazy_import, requires
from flash.core.utilities.providers import _FIFTYONE

if _FIFTYONE_AVAILABLE:
    fo = lazy_import("fiftyone")
    Detections = "fiftyone.Detections"
else:
    fo = None
    Detections = None


OBJECT_DETECTION_OUTPUTS = FlashRegistry("outputs")


@OBJECT_DETECTION_OUTPUTS(name="fiftyone", providers=_FIFTYONE)
class FiftyOneDetectionLabelsOutput(Output):
    """A :class:`.Output` which converts model outputs to FiftyOne detection format.

    Args:
        labels: A list of labels, assumed to map the class index to the label for that class.
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
        return_filepath: bool = True,
    ):
        super().__init__()
        self._labels = labels
        self.threshold = threshold
        self.return_filepath = return_filepath

    @classmethod
    def from_task(cls, task: Task, **kwargs) -> Output:
        return cls(labels=getattr(task, "labels", None))

    def transform(self, sample: Dict[str, Any]) -> Union[Detections, Dict[str, Any]]:
        if DataKeys.METADATA not in sample:
            raise ValueError("sample requires DataKeys.METADATA to use a FiftyOneDetectionLabelsOutput output.")

        height, width = sample[DataKeys.METADATA]["output_size"]

        detections = []

        preds = sample[DataKeys.PREDS]

        for bbox, label, score in zip(preds["bboxes"], preds["labels"], preds["scores"]):
            confidence = score.tolist()

            if self.threshold is not None and confidence < self.threshold:
                continue

            xmin, ymin, box_width, box_height = bbox["xmin"], bbox["ymin"], bbox["width"], bbox["height"]
            box = [
                (xmin / width).item(),
                (ymin / height).item(),
                (box_width / width).item(),
                (box_height / height).item(),
            ]

            label = label.item()
            if self._labels is not None:
                label = self._labels[label]
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
            filepath = sample[DataKeys.METADATA]["filepath"]
            return {"filepath": filepath, "predictions": fo_predictions}
        return fo_predictions
