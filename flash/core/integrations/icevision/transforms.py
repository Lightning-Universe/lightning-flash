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
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from torch import nn, Tensor

from flash.core.data.io.input import DataKeys
from flash.core.data.io.input_transform import InputTransform
from flash.core.utilities.imports import (
    _ICEVISION_AVAILABLE,
    _ICEVISION_GREATER_EQUAL_0_11_0,
    _IMAGE_AVAILABLE,
    requires,
)

if _IMAGE_AVAILABLE:
    from PIL import Image

if _ICEVISION_AVAILABLE:
    from icevision.core import tasks
    from icevision.core.bbox import BBox
    from icevision.core.keypoints import KeyPoints
    from icevision.core.mask import Mask, MaskArray
    from icevision.core.record import BaseRecord
    from icevision.core.record_components import (
        BBoxesRecordComponent,
        ClassMapRecordComponent,
        FilepathRecordComponent,
        ImageRecordComponent,
        InstancesLabelsRecordComponent,
        KeyPointsRecordComponent,
        RecordComponent,
        RecordIDRecordComponent,
    )
    from icevision.data.prediction import Prediction
    from icevision.tfms import A
else:
    MaskArray = object
    RecordComponent = object

    class tasks:
        common = object


if _ICEVISION_AVAILABLE and _ICEVISION_GREATER_EQUAL_0_11_0:
    from icevision.core.record_components import InstanceMasksRecordComponent
elif _ICEVISION_AVAILABLE:
    from icevision.core.record_components import MasksRecordComponent


def _split_mask_array(mask_array: MaskArray) -> List[MaskArray]:
    """Utility to split a single ``MaskArray`` object into a list of ``MaskArray`` objects (one per mask)."""
    return [MaskArray(mask) for mask in mask_array.data]


class OriginalSizeRecordComponent(RecordComponent):
    def __init__(self, original_size: Optional[Tuple[int, int]], task=tasks.common):
        super().__init__(task=task)
        # original_size: (h, w)
        self.original_size: Optional[Tuple[int, int]] = original_size


def to_icevision_record(sample: Dict[str, Any]):
    record = BaseRecord([])

    metadata = sample.get(DataKeys.METADATA, None) or {}

    if "image_id" in metadata:
        record_id_component = RecordIDRecordComponent()
        record_id_component.set_record_id(metadata["image_id"])

    component = ClassMapRecordComponent(tasks.detection)
    component.set_class_map(metadata.get("class_map", None))
    record.add_component(component)

    if isinstance(sample[DataKeys.INPUT], str):
        input_component = FilepathRecordComponent()
        input_component.set_filepath(sample[DataKeys.INPUT])
    else:
        if "filepath" in metadata:
            input_component = FilepathRecordComponent()
            input_component.filepath = metadata["filepath"]
        else:
            input_component = ImageRecordComponent()
        input_component.composite = record
        image = sample[DataKeys.INPUT]
        if isinstance(image, Tensor):
            image = image.permute(1, 2, 0).numpy()
        elif isinstance(image, Image.Image):
            image = np.array(image)
        input_component.set_img(image)

        record.add_component(OriginalSizeRecordComponent(metadata.get("size", image.shape[:2])))
    record.add_component(input_component)

    if DataKeys.TARGET in sample:
        if "labels" in sample[DataKeys.TARGET]:
            labels_component = InstancesLabelsRecordComponent()
            labels_component.add_labels_by_id(sample[DataKeys.TARGET]["labels"])
            record.add_component(labels_component)

        if "bboxes" in sample[DataKeys.TARGET]:
            bboxes = [
                BBox.from_xywh(bbox["xmin"], bbox["ymin"], bbox["width"], bbox["height"])
                for bbox in sample[DataKeys.TARGET]["bboxes"]
            ]
            bboxes_component = BBoxesRecordComponent()
            bboxes_component.set_bboxes(bboxes)
            record.add_component(bboxes_component)

        if _ICEVISION_GREATER_EQUAL_0_11_0:
            masks = sample[DataKeys.TARGET].get("masks", None)

            if masks is not None:
                component = InstanceMasksRecordComponent()

                if len(masks) > 0 and isinstance(masks[0], Mask):
                    component.set_masks(masks)
                else:
                    # TODO: This treats invalid examples as negative examples
                    if len(masks) == 0 or not (
                        len(masks) == len(record.detection.bboxes) == len(record.detection.label_ids)
                    ):
                        data = np.zeros((0, record.height, record.width), np.uint8)
                        labels_component.label_ids = []
                        bboxes_component.bboxes = []
                    else:
                        data = np.stack(masks, axis=0)
                    mask_array = MaskArray(data)
                    component.set_mask_array(mask_array)
                    component.set_masks(_split_mask_array(mask_array))

                record.add_component(component)
        else:
            mask_array = sample[DataKeys.TARGET].get("mask_array", None)
            if mask_array is not None:
                component = MasksRecordComponent()
                component.set_masks(mask_array)
                record.add_component(component)

        if "keypoints" in sample[DataKeys.TARGET]:
            keypoints = []

            keypoints_metadata = sample[DataKeys.TARGET].get(
                "keypoints_metadata", [None] * len(sample[DataKeys.TARGET]["keypoints"])
            )

            for keypoints_list, keypoints_metadata in zip(sample[DataKeys.TARGET]["keypoints"], keypoints_metadata):
                xyv = []
                for keypoint in keypoints_list:
                    xyv.extend((keypoint["x"], keypoint["y"], keypoint["visible"]))

                keypoints.append(KeyPoints.from_xyv(xyv, keypoints_metadata))
            component = KeyPointsRecordComponent()
            component.set_keypoints(keypoints)
            record.add_component(component)

    return record


def from_icevision_detection(record: "BaseRecord"):
    detection = record.detection

    result = {}

    if hasattr(detection, "bboxes"):
        result["bboxes"] = [
            {
                "xmin": bbox.xmin,
                "ymin": bbox.ymin,
                "width": bbox.width,
                "height": bbox.height,
            }
            for bbox in detection.bboxes
        ]

    masks = getattr(detection, "masks", None)
    mask_array = getattr(detection, "mask_array", None)
    if mask_array is not None or not _ICEVISION_GREATER_EQUAL_0_11_0:
        if not isinstance(mask_array, MaskArray) or len(mask_array.data) == 0:
            mask_array = MaskArray.from_masks(masks, record.height, record.width)

        result["masks"] = [mask.data[0] for mask in _split_mask_array(mask_array)]
    elif masks is not None:
        result["masks"] = masks  # Note - this doesn't unpack IceVision objects

    if hasattr(detection, "keypoints"):
        keypoints = detection.keypoints

        result["keypoints"] = []
        result["keypoints_metadata"] = []

        for keypoint in keypoints:
            keypoints_list = []
            for x, y, v in keypoint.xyv:
                keypoints_list.append(
                    {
                        "x": x,
                        "y": y,
                        "visible": v,
                    }
                )
            result["keypoints"].append(keypoints_list)

            # TODO: Unpack keypoints_metadata
            result["keypoints_metadata"].append(keypoint.metadata)

    if getattr(detection, "label_ids", None) is not None:
        result["labels"] = list(detection.label_ids)

    if getattr(detection, "scores", None) is not None:
        result["scores"] = list(detection.scores)

    return result


def from_icevision_record(record: "BaseRecord"):
    sample = {
        DataKeys.METADATA: {
            "size": getattr(record, "original_size", (record.height, record.width)),
            "output_size": (record.height, record.width),
        }
    }

    if getattr(record, "record_id", None) is not None:
        sample[DataKeys.METADATA]["image_id"] = record.record_id

    if getattr(record, "filepath", None) is not None:
        sample[DataKeys.METADATA]["filepath"] = record.filepath

    if record.img is not None:
        sample[DataKeys.INPUT] = record.img
        filepath = getattr(record, "filepath", None)
        if filepath is not None:
            sample[DataKeys.METADATA]["filepath"] = filepath
    elif getattr(record, "filepath", None) is not None:
        sample[DataKeys.INPUT] = record.filepath

    sample[DataKeys.TARGET] = from_icevision_detection(record)

    if getattr(record.detection, "class_map", None) is not None:
        sample[DataKeys.METADATA]["class_map"] = record.detection.class_map

    return sample


def from_icevision_predictions(predictions: List["Prediction"]):
    result = []
    for prediction in predictions:
        result.append(from_icevision_detection(prediction.pred))
    return result


class IceVisionTransformAdapter(nn.Module):
    """
    Args:
        transform: list of transformation functions to apply

    """

    def __init__(self, transform: List[Callable]):
        super().__init__()
        self.transform = A.Adapter(transform)

    def forward(self, x):
        record = to_icevision_record(x)
        record = self.transform(record)
        return from_icevision_record(record)


@dataclass
class IceVisionInputTransform(InputTransform):

    image_size: int = 128

    @requires("image", "icevision")
    def per_sample_transform(self):
        return IceVisionTransformAdapter([*A.resize_and_pad(self.image_size), A.Normalize()])

    @requires("image", "icevision")
    def train_per_sample_transform(self):
        return IceVisionTransformAdapter([*A.aug_tfms(size=self.image_size), A.Normalize()])

    def collate(self) -> Callable:
        return self._identity
