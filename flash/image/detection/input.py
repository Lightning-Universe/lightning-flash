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
from typing import Any, Dict, Hashable, List, Optional, Sequence

from flash.core.data.io.classification_input import ClassificationInputMixin
from flash.core.data.io.input import DataKeys
from flash.core.data.utilities.classification import TargetFormatter
from flash.core.data.utilities.paths import filter_valid_files, PATH_TYPE
from flash.core.data.utilities.samples import to_samples
from flash.core.integrations.fiftyone.utils import FiftyOneLabelUtilities
from flash.core.integrations.icevision.data import IceVisionInput
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE, _ICEVISION_AVAILABLE, lazy_import, requires
from flash.image.data import (
    ImageFilesInput,
    ImageInput,
    ImageNumpyInput,
    ImageTensorInput,
    IMG_EXTENSIONS,
    NP_EXTENSIONS,
)

if _FIFTYONE_AVAILABLE:
    fol = lazy_import("fiftyone.core.labels")
    SampleCollection = "fiftyone.core.collections.SampleCollection"
else:
    fol = None
    SampleCollection = None

if _ICEVISION_AVAILABLE:
    from icevision.core import BBox, ClassMap, IsCrowdsRecordComponent, ObjectDetectionRecord
    from icevision.data import SingleSplitSplitter
    from icevision.parsers import Parser
    from icevision.utils import ImgSize
else:
    Parser = object


class ObjectDetectionFilesInput(ClassificationInputMixin, ImageFilesInput):
    def load_data(
        self,
        files: List[PATH_TYPE],
        targets: Optional[List[List[Any]]] = None,
        bboxes: Optional[List[List[Dict[str, int]]]] = None,
        target_formatter: Optional[TargetFormatter] = None,
    ) -> List[Dict[str, Any]]:
        if targets is None:
            return super().load_data(files)
        files, targets, bboxes = filter_valid_files(
            files, targets, bboxes, valid_extensions=IMG_EXTENSIONS + NP_EXTENSIONS
        )
        self.load_target_metadata(
            [t for target in targets for t in target], add_background=True, target_formatter=target_formatter
        )

        return [
            {DataKeys.INPUT: file, DataKeys.TARGET: {"bboxes": bbox, "labels": label}}
            for file, label, bbox in zip(files, targets, bboxes)
        ]

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample = super().load_sample(sample)
        if DataKeys.TARGET in sample:
            sample[DataKeys.TARGET]["labels"] = [
                self.format_target(label) for label in sample[DataKeys.TARGET]["labels"]
            ]
        return sample


class ObjectDetectionNumpyInput(ClassificationInputMixin, ImageNumpyInput):
    def load_data(
        self,
        array: Any,
        targets: Optional[List[List[Any]]] = None,
        bboxes: Optional[List[List[Dict[str, int]]]] = None,
        target_formatter: Optional[TargetFormatter] = None,
    ) -> List[Dict[str, Any]]:
        if targets is None:
            return to_samples(array)
        self.load_target_metadata(
            [t for target in targets for t in target], add_background=True, target_formatter=target_formatter
        )

        return [
            {DataKeys.INPUT: image, DataKeys.TARGET: {"bboxes": bbox, "labels": label}}
            for image, label, bbox in zip(array, targets, bboxes)
        ]

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample = super().load_sample(sample)
        if DataKeys.TARGET in sample:
            sample[DataKeys.TARGET]["labels"] = [
                self.format_target(label) for label in sample[DataKeys.TARGET]["labels"]
            ]
        return sample


class ObjectDetectionImageInput(ClassificationInputMixin, ImageInput):
    def load_data(
        self,
        images: Any,
        targets: Optional[List[List[Any]]] = None,
        bboxes: Optional[List[List[Dict[str, int]]]] = None,
        target_formatter: Optional[TargetFormatter] = None,
    ) -> List[Dict[str, Any]]:
        if targets is None:
            return to_samples(images)
        self.load_target_metadata(
            [t for target in targets for t in target], add_background=True, target_formatter=target_formatter
        )

        return [
            {DataKeys.INPUT: image, DataKeys.TARGET: {"bboxes": bbox, "labels": label}}
            for image, label, bbox in zip(images, targets, bboxes)
        ]

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample = super().load_sample(sample)
        if DataKeys.TARGET in sample:
            sample[DataKeys.TARGET]["labels"] = [
                self.format_target(label) for label in sample[DataKeys.TARGET]["labels"]
            ]
        return sample


class ObjectDetectionTensorInput(ClassificationInputMixin, ImageTensorInput):
    def load_data(
        self,
        tensor: Any,
        targets: Optional[List[List[Any]]] = None,
        bboxes: Optional[List[List[Dict[str, int]]]] = None,
        target_formatter: Optional[TargetFormatter] = None,
    ) -> List[Dict[str, Any]]:
        if targets is None:
            return to_samples(tensor)
        self.load_target_metadata(
            [t for target in targets for t in target], add_background=True, target_formatter=target_formatter
        )

        return [
            {DataKeys.INPUT: image, DataKeys.TARGET: {"bboxes": bbox, "labels": label}}
            for image, label, bbox in zip(tensor, targets, bboxes)
        ]

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample = super().load_sample(sample)
        if DataKeys.TARGET in sample:
            sample[DataKeys.TARGET]["labels"] = [
                self.format_target(label) for label in sample[DataKeys.TARGET]["labels"]
            ]
        return sample


class FiftyOneParser(Parser):
    def __init__(self, data, class_map, label_field, iscrowd):
        template_record = ObjectDetectionRecord()
        template_record.add_component(IsCrowdsRecordComponent())
        super().__init__(template_record=template_record)

        data = data
        label_field = label_field
        iscrowd = iscrowd

        self.data = []
        self.class_map = class_map

        for fp, w, h, sample_labs, sample_boxes, sample_iscrowd in zip(
            data.values("filepath"),
            data.values("metadata.width"),
            data.values("metadata.height"),
            data.values(label_field + ".detections.label"),
            data.values(label_field + ".detections.bounding_box"),
            data.values(label_field + ".detections." + iscrowd),
        ):
            for lab, box, iscrowd in zip(sample_labs, sample_boxes, sample_iscrowd):
                self.data.append((fp, w, h, lab, box, iscrowd))

    def __iter__(self) -> Any:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def record_id(self, o) -> Hashable:
        return o[0]

    def parse_fields(self, o, record, is_new):
        fp, w, h, lab, box, iscrowd = o

        if iscrowd is None:
            iscrowd = 0

        if is_new:
            record.set_filepath(fp)
            record.set_img_size(ImgSize(width=w, height=h))
            record.detection.set_class_map(self.class_map)

        box = self._reformat_bbox(*box, w, h)

        record.detection.add_bboxes([BBox.from_xyxy(*box)])
        record.detection.add_labels([lab])
        record.detection.add_iscrowds([iscrowd])

    @staticmethod
    def _reformat_bbox(xmin, ymin, box_w, box_h, img_w, img_h):
        xmin *= img_w
        ymin *= img_h
        box_w *= img_w
        box_h *= img_h
        xmax = xmin + box_w
        ymax = ymin + box_h
        output_bbox = [xmin, ymin, xmax, ymax]
        return output_bbox


class ObjectDetectionFiftyOneInput(IceVisionInput):
    num_classes: int
    labels: list

    @requires("fiftyone")
    def load_data(
        self,
        sample_collection: SampleCollection,
        label_field: str = "ground_truth",
        iscrowd: str = "iscrowd",
    ) -> Sequence[Dict[str, Any]]:
        label_utilities = FiftyOneLabelUtilities(label_field, fol.Detections)
        label_utilities.validate(sample_collection)
        sample_collection.compute_metadata()
        classes = label_utilities.get_classes(sample_collection)
        class_map = ClassMap(classes)
        self.num_classes = len(class_map)
        self.labels = [class_map.get_by_id(i) for i in range(self.num_classes)]

        parser = FiftyOneParser(sample_collection, class_map, label_field, iscrowd)
        records = parser.parse(data_splitter=SingleSplitSplitter())
        return [{DataKeys.INPUT: record} for record in records[0]]

    @staticmethod
    @requires("fiftyone")
    def predict_load_data(sample_collection: SampleCollection) -> Sequence[Dict[str, Any]]:
        return [{DataKeys.INPUT: f} for f in sample_collection.values("filepath")]
