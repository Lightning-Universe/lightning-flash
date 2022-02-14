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
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from flash.core.data.io.input import DataKeys, Input
from flash.core.data.utilities.paths import filter_valid_files, PATH_TYPE
from flash.core.data.utilities.samples import to_samples
from flash.core.integrations.fiftyone.utils import FiftyOneLabelUtilities
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE, _TORCHVISION_AVAILABLE, lazy_import
from flash.image.data import image_loader, ImageDeserializer, IMG_EXTENSIONS
from flash.image.segmentation.output import SegmentationLabelsOutput

if _FIFTYONE_AVAILABLE:
    fo = lazy_import("fiftyone")
    SampleCollection = "fiftyone.core.collections.SampleCollection"
else:
    fo = None
    SampleCollection = None

if _TORCHVISION_AVAILABLE:
    from torchvision.transforms.functional import to_tensor


class SemanticSegmentationInput(Input):
    def load_labels_map(
        self, num_classes: Optional[int] = None, labels_map: Optional[Dict[int, Tuple[int, int, int]]] = None
    ) -> None:
        if num_classes is not None:
            self.num_classes = num_classes
            labels_map = labels_map or SegmentationLabelsOutput.create_random_labels_map(num_classes)

        if labels_map is not None:
            self.labels_map = labels_map

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample[DataKeys.INPUT] = sample[DataKeys.INPUT].float()
        if DataKeys.TARGET in sample:
            sample[DataKeys.TARGET] = sample[DataKeys.TARGET].float()
        sample[DataKeys.METADATA] = {"size": sample[DataKeys.INPUT].shape[-2:]}
        return sample


class SemanticSegmentationTensorInput(SemanticSegmentationInput):
    def load_data(
        self,
        tensor: Any,
        masks: Any = None,
        num_classes: Optional[int] = None,
        labels_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ) -> List[Dict[str, Any]]:
        self.load_labels_map(num_classes, labels_map)
        return to_samples(tensor, masks)


class SemanticSegmentationNumpyInput(SemanticSegmentationInput):
    def load_data(
        self,
        array: Any,
        masks: Any = None,
        num_classes: Optional[int] = None,
        labels_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ) -> List[Dict[str, Any]]:
        self.load_labels_map(num_classes, labels_map)
        return to_samples(array, masks)

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample[DataKeys.INPUT] = torch.from_numpy(sample[DataKeys.INPUT])
        if DataKeys.TARGET in sample:
            sample[DataKeys.TARGET] = torch.from_numpy(sample[DataKeys.TARGET])
        return super().load_sample(sample)


class SemanticSegmentationFilesInput(SemanticSegmentationInput):
    def load_data(
        self,
        files: Union[PATH_TYPE, List[PATH_TYPE]],
        mask_files: Optional[Union[PATH_TYPE, List[PATH_TYPE]]] = None,
        num_classes: Optional[int] = None,
        labels_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ) -> List[Dict[str, Any]]:
        self.load_labels_map(num_classes, labels_map)
        if mask_files is None:
            files = filter_valid_files(files, valid_extensions=IMG_EXTENSIONS)
        else:
            files, mask_files = filter_valid_files(files, mask_files, valid_extensions=IMG_EXTENSIONS)
        return to_samples(files, mask_files)

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        filepath = sample[DataKeys.INPUT]
        sample[DataKeys.INPUT] = to_tensor(image_loader(filepath))
        if DataKeys.TARGET in sample:
            sample[DataKeys.TARGET] = (to_tensor(image_loader(sample[DataKeys.TARGET])) * 255).long()[0]
        sample = super().load_sample(sample)
        sample[DataKeys.METADATA]["filepath"] = filepath
        return sample


class SemanticSegmentationFolderInput(SemanticSegmentationFilesInput):
    def load_data(
        self,
        folder: PATH_TYPE,
        mask_folder: Optional[PATH_TYPE] = None,
        num_classes: Optional[int] = None,
        labels_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ) -> List[Dict[str, Any]]:
        self.load_labels_map(num_classes, labels_map)
        files = os.listdir(folder)
        files.sort()
        if mask_folder is not None:
            mask_files = {os.path.splitext(file)[0]: file for file in os.listdir(mask_folder)}
            file_names = [os.path.splitext(file)[0] for file in files]

            if len(set(file_names) - mask_files.keys()) != 0:
                raise ValueError(
                    f"Found inconsistent files in input folder: {folder} and mask folder: {mask_folder}. All input "
                    f"files must have a corresponding mask file with the same name."
                )

            files = [os.path.join(folder, file) for file in files]
            mask_files = [os.path.join(mask_folder, mask_files[file_name]) for file_name in file_names]
            return super().load_data(files, mask_files)
        return super().load_data([os.path.join(folder, file) for file in files])


class SemanticSegmentationFiftyOneInput(SemanticSegmentationFilesInput):
    def load_data(
        self,
        sample_collection: SampleCollection,
        label_field: str = "ground_truth",
        num_classes: Optional[int] = None,
        labels_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ) -> List[Dict[str, Any]]:
        self.load_labels_map(num_classes, labels_map)

        self.label_field = label_field
        label_utilities = FiftyOneLabelUtilities(label_field, fo.Segmentation)
        label_utilities.validate(sample_collection)

        self._fo_dataset_name = sample_collection.name
        return to_samples(sample_collection.values("filepath"))

    def predict_load_data(
        self,
        sample_collection: SampleCollection,
    ) -> List[Dict[str, Any]]:
        return to_samples(sample_collection.values("filepath"))

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        filepath = sample[DataKeys.INPUT]
        sample = super().load_sample(sample)
        if not self.predicting:
            fo_dataset = fo.load_dataset(self._fo_dataset_name)
            fo_sample = fo_dataset[filepath]
            sample[DataKeys.TARGET] = torch.from_numpy(fo_sample[self.label_field].mask).float()
        return sample


class SemanticSegmentationDeserializer(ImageDeserializer):
    def serve_load_sample(self, data: str) -> Dict[str, Any]:
        result = super().serve_load_sample(data)
        result[DataKeys.INPUT] = to_tensor(result[DataKeys.INPUT])
        result[DataKeys.METADATA] = {"size": result[DataKeys.INPUT].shape[-2:]}
        return result
