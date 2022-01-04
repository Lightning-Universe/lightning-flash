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
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import torch
from pytorch_lightning.utilities import rank_zero_warn

from flash.core.data.io.input import DataKeys, ImageLabelsMap, Input
from flash.core.data.utilities.paths import filter_valid_files, PATH_TYPE
from flash.core.data.utilities.samples import to_samples
from flash.core.data.utils import image_default_loader
from flash.core.integrations.fiftyone.utils import FiftyOneLabelUtilities
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE, _TORCHVISION_AVAILABLE, lazy_import
from flash.image.data import ImageDeserializer, IMG_EXTENSIONS
from flash.image.segmentation.output import SegmentationLabelsOutput

SampleCollection = None
if _FIFTYONE_AVAILABLE:
    fo = lazy_import("fiftyone")
    if TYPE_CHECKING:
        from fiftyone.core.collections import SampleCollection
else:
    fo = None

if _TORCHVISION_AVAILABLE:
    import torchvision
    import torchvision.transforms.functional as FT


class SemanticSegmentationInput(Input):
    def load_labels_map(
        self, num_classes: Optional[int] = None, labels_map: Optional[Dict[int, Tuple[int, int, int]]] = None
    ) -> None:
        if num_classes is not None:
            self.num_classes = num_classes
            labels_map = labels_map or SegmentationLabelsOutput.create_random_labels_map(num_classes)

        if labels_map is not None:
            self.set_state(ImageLabelsMap(labels_map))
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
            files, masks = filter_valid_files(files, mask_files, valid_extensions=IMG_EXTENSIONS)
        return to_samples(files, mask_files)

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        filepath = sample[DataKeys.INPUT]
        sample[DataKeys.INPUT] = FT.to_tensor(image_default_loader(filepath))
        if DataKeys.TARGET in sample:
            sample[DataKeys.TARGET] = torchvision.io.read_image(sample[DataKeys.TARGET])[0]
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
            mask_files = os.listdir(mask_folder)

            all_files = set(files).intersection(set(mask_files))
            if len(all_files) != len(files) or len(all_files) != len(mask_files):
                rank_zero_warn(
                    f"Found inconsistent files in input folder: {folder} and mask folder: {mask_folder}. Some files"
                    " have been dropped.",
                    UserWarning,
                )

            files = [os.path.join(folder, file) for file in all_files]
            mask_files = [os.path.join(mask_folder, file) for file in all_files]
            files.sort()
            mask_files.sort()
            return super().load_data(files, mask_files)
        return super().load_data(files)


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

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        filepath = sample[DataKeys.INPUT]
        sample = super().load_sample(sample)
        if not self.predicting:
            fo_dataset = fo.load_dataset(self._fo_dataset_name)
            fo_sample = fo_dataset[filepath]
            sample[DataKeys.TARGET] = torch.from_numpy(fo_sample[self.label_field].mask).float()  # H x W
        return sample


class SemanticSegmentationDeserializer(ImageDeserializer):
    def serve_load_sample(self, data: str) -> Dict[str, Any]:
        result = super().serve_load_sample(data)
        result[DataKeys.INPUT] = FT.to_tensor(result[DataKeys.INPUT])
        result[DataKeys.METADATA] = {"size": result[DataKeys.INPUT].shape[-2:]}
        return result
