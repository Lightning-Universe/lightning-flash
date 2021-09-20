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
from typing import Any, Callable, Dict, List, Tuple

from torch import nn

from flash.core.data.data_source import DefaultDataKeys
from flash.core.utilities.imports import _ICEVISION_AVAILABLE, requires

if _ICEVISION_AVAILABLE:
    from icevision.core import tasks
    from icevision.core.bbox import BBox
    from icevision.core.keypoints import KeyPoints
    from icevision.core.mask import EncodedRLEs, MaskArray
    from icevision.core.record import BaseRecord
    from icevision.core.record_components import (
        BBoxesRecordComponent,
        ClassMapRecordComponent,
        FilepathRecordComponent,
        ImageRecordComponent,
        InstancesLabelsRecordComponent,
        KeyPointsRecordComponent,
        MasksRecordComponent,
        RecordIDRecordComponent,
    )
    from icevision.data.prediction import Prediction
    from icevision.tfms import A


def to_icevision_record(sample: Dict[str, Any]):
    record = BaseRecord([])

    metadata = sample.get(DefaultDataKeys.METADATA, None) or {}

    if "image_id" in metadata:
        record_id_component = RecordIDRecordComponent()
        record_id_component.set_record_id(metadata["image_id"])

    component = ClassMapRecordComponent(tasks.detection)
    component.set_class_map(metadata.get("class_map", None))
    record.add_component(component)

    if "labels" in sample[DefaultDataKeys.TARGET]:
        labels_component = InstancesLabelsRecordComponent()
        labels_component.add_labels_by_id(sample[DefaultDataKeys.TARGET]["labels"])
        record.add_component(labels_component)

    if "bboxes" in sample[DefaultDataKeys.TARGET]:
        bboxes = [
            BBox.from_xywh(bbox["xmin"], bbox["ymin"], bbox["width"], bbox["height"])
            for bbox in sample[DefaultDataKeys.TARGET]["bboxes"]
        ]
        component = BBoxesRecordComponent()
        component.set_bboxes(bboxes)
        record.add_component(component)

    if "masks" in sample[DefaultDataKeys.TARGET]:
        mask_array = MaskArray(sample[DefaultDataKeys.TARGET]["masks"])
        component = MasksRecordComponent()
        component.set_masks(mask_array)
        record.add_component(component)

    if "keypoints" in sample[DefaultDataKeys.TARGET]:
        keypoints = []

        for keypoints_list, keypoints_metadata in zip(
            sample[DefaultDataKeys.TARGET]["keypoints"], sample[DefaultDataKeys.TARGET]["keypoints_metadata"]
        ):
            xyv = []
            for keypoint in keypoints_list:
                xyv.extend((keypoint["x"], keypoint["y"], keypoint["visible"]))

            keypoints.append(KeyPoints.from_xyv(xyv, keypoints_metadata))
        component = KeyPointsRecordComponent()
        component.set_keypoints(keypoints)
        record.add_component(component)

    if isinstance(sample[DefaultDataKeys.INPUT], str):
        input_component = FilepathRecordComponent()
        input_component.set_filepath(sample[DefaultDataKeys.INPUT])
    else:
        if "filepath" in metadata:
            input_component = FilepathRecordComponent()
            input_component.filepath = metadata["filepath"]
        else:
            input_component = ImageRecordComponent()
        input_component.composite = record
        input_component.set_img(sample[DefaultDataKeys.INPUT])
    record.add_component(input_component)

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

    if hasattr(detection, "masks"):
        masks = detection.masks

        if isinstance(masks, EncodedRLEs):
            masks = masks.to_mask(record.height, record.width)

        if isinstance(masks, MaskArray):
            result["masks"] = masks.data
        else:
            raise RuntimeError("Masks are expected to be a MaskArray or EncodedRLEs.")

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
        DefaultDataKeys.METADATA: {
            "size": (record.height, record.width),
        }
    }

    if getattr(record, "record_id", None) is not None:
        sample[DefaultDataKeys.METADATA]["image_id"] = record.record_id

    if getattr(record, "filepath", None) is not None:
        sample[DefaultDataKeys.METADATA]["filepath"] = record.filepath

    if record.img is not None:
        sample[DefaultDataKeys.INPUT] = record.img
        filepath = getattr(record, "filepath", None)
        if filepath is not None:
            sample[DefaultDataKeys.METADATA]["filepath"] = filepath
    elif record.filepath is not None:
        sample[DefaultDataKeys.INPUT] = record.filepath

    sample[DefaultDataKeys.TARGET] = from_icevision_detection(record)

    if getattr(record.detection, "class_map", None) is not None:
        sample[DefaultDataKeys.METADATA]["class_map"] = record.detection.class_map

    return sample


def from_icevision_predictions(predictions: List["Prediction"]):
    result = []
    for prediction in predictions:
        result.append(from_icevision_detection(prediction.pred))
    return result


class IceVisionTransformAdapter(nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = A.Adapter(transform)

    def forward(self, x):
        record = to_icevision_record(x)
        record = self.transform(record)
        return from_icevision_record(record)


@requires(["image", "icevision"])
def default_transforms(image_size: Tuple[int, int]) -> Dict[str, Callable]:
    """The default transforms from IceVision."""
    return {
        "pre_tensor_transform": IceVisionTransformAdapter([*A.resize_and_pad(image_size), A.Normalize()]),
    }


@requires(["image", "icevision"])
def train_default_transforms(image_size: Tuple[int, int]) -> Dict[str, Callable]:
    """The default augmentations from IceVision."""
    return {
        "pre_tensor_transform": IceVisionTransformAdapter([*A.aug_tfms(size=image_size), A.Normalize()]),
    }
