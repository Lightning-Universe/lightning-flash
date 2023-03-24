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
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from flash.core.data.io.classification_input import ClassificationInputMixin
from flash.core.data.io.input import DataKeys
from flash.core.data.utilities.classification import MultiBinaryTargetFormatter, TargetFormatter
from flash.core.data.utilities.data_frame import resolve_files, resolve_targets
from flash.core.data.utilities.loading import load_data_frame
from flash.core.data.utilities.paths import filter_valid_files, make_dataset, PATH_TYPE
from flash.core.data.utilities.samples import to_samples
from flash.core.integrations.fiftyone.utils import FiftyOneLabelUtilities
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE, lazy_import, requires
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


class ImageClassificationFilesInput(ClassificationInputMixin, ImageFilesInput):
    def load_data(
        self,
        files: List[PATH_TYPE],
        targets: Optional[List[Any]] = None,
        target_formatter: Optional[TargetFormatter] = None,
    ) -> List[Dict[str, Any]]:
        if targets is None:
            return super().load_data(files)
        files, targets = filter_valid_files(files, targets, valid_extensions=IMG_EXTENSIONS + NP_EXTENSIONS)
        self.load_target_metadata(targets, target_formatter=target_formatter)
        return to_samples(files, targets)

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample = super().load_sample(sample)
        if DataKeys.TARGET in sample:
            sample[DataKeys.TARGET] = self.format_target(sample[DataKeys.TARGET])
        return sample


class ImageClassificationFolderInput(ImageClassificationFilesInput):
    def load_data(self, folder: PATH_TYPE, target_formatter: Optional[TargetFormatter] = None) -> List[Dict[str, Any]]:
        files, targets = make_dataset(folder, extensions=IMG_EXTENSIONS + NP_EXTENSIONS)
        return super().load_data(files, targets, target_formatter=target_formatter)


class ImageClassificationFiftyOneInput(ImageClassificationFilesInput):
    @requires("fiftyone")
    def load_data(
        self,
        sample_collection: SampleCollection,
        label_field: str = "ground_truth",
        target_formatter: Optional[TargetFormatter] = None,
    ) -> List[Dict[str, Any]]:
        label_utilities = FiftyOneLabelUtilities(label_field, fol.Label)
        label_utilities.validate(sample_collection)

        label_path = sample_collection._get_label_field_path(label_field, "label")[1]

        filepaths = sample_collection.values("filepath")
        targets = sample_collection.values(label_path)

        return super().load_data(filepaths, targets, target_formatter=target_formatter)

    @requires("fiftyone")
    def predict_load_data(
        self, data: SampleCollection, target_formatter: Optional[TargetFormatter] = None
    ) -> List[Dict[str, Any]]:
        return super().load_data(data.values("filepath"), target_formatter=target_formatter)


class ImageClassificationTensorInput(ClassificationInputMixin, ImageTensorInput):
    def load_data(
        self, tensor: Any, targets: Optional[List[Any]] = None, target_formatter: Optional[TargetFormatter] = None
    ) -> List[Dict[str, Any]]:
        if targets is not None:
            self.load_target_metadata(targets, target_formatter=target_formatter)
        return to_samples(tensor, targets)

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample = super().load_sample(sample)
        if DataKeys.TARGET in sample:
            sample[DataKeys.TARGET] = self.format_target(sample[DataKeys.TARGET])
        return sample


class ImageClassificationNumpyInput(ClassificationInputMixin, ImageNumpyInput):
    def load_data(
        self, array: Any, targets: Optional[List[Any]] = None, target_formatter: Optional[TargetFormatter] = None
    ) -> List[Dict[str, Any]]:
        if targets is not None:
            self.load_target_metadata(targets, target_formatter=target_formatter)
        return to_samples(array, targets)

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample = super().load_sample(sample)
        if DataKeys.TARGET in sample:
            sample[DataKeys.TARGET] = self.format_target(sample[DataKeys.TARGET])
        return sample


class ImageClassificationImageInput(ClassificationInputMixin, ImageInput):
    def load_data(
        self, images: Any, targets: Optional[List[Any]] = None, target_formatter: Optional[TargetFormatter] = None
    ) -> List[Dict[str, Any]]:
        if targets is not None:
            self.load_target_metadata(targets, target_formatter=target_formatter)
        return to_samples(images, targets)

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample = super().load_sample(sample)
        if DataKeys.TARGET in sample:
            sample[DataKeys.TARGET] = self.format_target(sample[DataKeys.TARGET])
        return sample


class ImageClassificationDataFrameInput(ImageClassificationFilesInput):
    labels: list

    def load_data(
        self,
        data_frame: pd.DataFrame,
        input_key: str,
        target_keys: Optional[Union[str, List[str]]] = None,
        root: Optional[PATH_TYPE] = None,
        resolver: Optional[Callable[[Optional[PATH_TYPE], Any], PATH_TYPE]] = None,
        target_formatter: Optional[TargetFormatter] = None,
    ) -> List[Dict[str, Any]]:
        files = resolve_files(data_frame, input_key, root, resolver)
        if target_keys is not None:
            targets = resolve_targets(data_frame, target_keys)
        else:
            targets = None
        result = super().load_data(files, targets, target_formatter=target_formatter)

        # If we had binary multi-class targets then we also know the labels (column names)
        if (
            self.training
            and isinstance(self.target_formatter, MultiBinaryTargetFormatter)
            and isinstance(target_keys, List)
        ):
            self.labels = target_keys

        return result


class ImageClassificationCSVInput(ImageClassificationDataFrameInput):
    def load_data(
        self,
        csv_file: PATH_TYPE,
        input_key: str,
        target_keys: Optional[Union[str, List[str]]] = None,
        root: Optional[PATH_TYPE] = None,
        resolver: Optional[Callable[[Optional[PATH_TYPE], Any], PATH_TYPE]] = None,
        target_formatter: Optional[TargetFormatter] = None,
    ) -> List[Dict[str, Any]]:
        data_frame = load_data_frame(csv_file)
        if root is None:
            root = os.path.dirname(csv_file)
        return super().load_data(data_frame, input_key, target_keys, root, resolver, target_formatter=target_formatter)
