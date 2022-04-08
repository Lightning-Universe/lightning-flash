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
from typing import Any, Dict, Optional, Sequence, Type

from torch.utils.data import Dataset

from flash.core.data.data_module import DataModule
from flash.core.data.io.input import Input
from flash.core.utilities.stability import beta
from flash.core.utilities.stages import RunningStage
from flash.core.utilities.types import INPUT_TRANSFORM_TYPE
from flash.image.classification.data import ImageClassificationFilesInput, ImageClassificationFolderInput
from flash.image.face_detection.input import FaceDetectionInput
from flash.image.face_detection.input_transform import FaceDetectionInputTransform


@beta("Face detection is currently in Beta.")
class FaceDetectionData(DataModule):
    input_transform_cls = FaceDetectionInputTransform

    @classmethod
    def from_datasets(
        cls,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        predict_dataset: Optional[Dataset] = None,
        input_cls: Type[Input] = FaceDetectionInput,
        transform: INPUT_TRANSFORM_TYPE = FaceDetectionInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs,
    ) -> "FaceDetectionData":

        ds_kw = dict()

        return cls(
            input_cls(RunningStage.TRAINING, train_dataset, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_dataset, **ds_kw),
            input_cls(RunningStage.TESTING, test_dataset, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_dataset, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_files(
        cls,
        predict_files: Optional[Sequence[str]] = None,
        predict_transform: INPUT_TRANSFORM_TYPE = FaceDetectionInputTransform,
        input_cls: Type[Input] = ImageClassificationFilesInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "FaceDetectionData":

        return cls(
            predict_input=input_cls(RunningStage.PREDICTING, predict_files),
            transform=predict_transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_folders(
        cls,
        predict_folder: Optional[str] = None,
        predict_transform: INPUT_TRANSFORM_TYPE = FaceDetectionInputTransform,
        input_cls: Type[Input] = ImageClassificationFolderInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "FaceDetectionData":

        return cls(
            predict_input=input_cls(RunningStage.PREDICTING, predict_folder),
            transform=predict_transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )
