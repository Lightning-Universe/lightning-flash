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
from typing import Any, Callable, Dict, Optional, Sequence, Union

from torch.utils.data import Dataset

from flash.audio.speech_recognition.input import (
    SpeechRecognitionCSVInput,
    SpeechRecognitionDatasetInput,
    SpeechRecognitionDeserializer,
    SpeechRecognitionJSONInput,
    SpeechRecognitionPathsInput,
)
from flash.audio.speech_recognition.output_transform import SpeechRecognitionOutputTransform
from flash.core.data.data_module import DataModule
from flash.core.utilities.stages import RunningStage


class SpeechRecognitionData(DataModule):
    """Data Module for text classification tasks."""

    input_transform_cls = SpeechRecognitionInputTransform
    output_transform_cls = SpeechRecognitionOutputTransform

    @classmethod
    def from_files(
        cls,
        train_files: Optional[Sequence[str]] = None,
        train_targets: Optional[Sequence[Any]] = None,
        val_files: Optional[Sequence[str]] = None,
        val_targets: Optional[Sequence[Any]] = None,
        test_files: Optional[Sequence[str]] = None,
        test_targets: Optional[Sequence[Any]] = None,
        predict_files: Optional[Sequence[str]] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        sampling_rate: int = 16000,
        **data_module_kwargs,
    ) -> "SpeechRecognitionData":
        return cls(
            SpeechRecognitionPathsInput(RunningStage.TRAINING, train_files, train_targets, sampling_rate=sampling_rate),
            SpeechRecognitionPathsInput(RunningStage.VALIDATING, val_files, val_targets, sampling_rate=sampling_rate),
            SpeechRecognitionPathsInput(RunningStage.TESTING, test_files, test_targets, sampling_rate=sampling_rate),
            SpeechRecognitionPathsInput(RunningStage.PREDICTING, predict_files, sampling_rate=sampling_rate),
            input_transform=cls.input_transform_cls(
                train_transform, val_transform, test_transform, predict_transform, sampling_rate
            ),
            **data_module_kwargs,
        )

    @classmethod
    def from_csv(
        cls,
        input_fields: Union[str, Sequence[str]],
        target_fields: Optional[str] = None,
        train_file: Optional[str] = None,
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        predict_file: Optional[str] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        sampling_rate: int = 16000,
        **data_module_kwargs: Any,
    ) -> "SpeechRecognitionData":
        dataset_kwargs = dict(
            input_key=input_fields,
            target_key=target_fields,
            sampling_rate=sampling_rate,
        )
        return cls(
            SpeechRecognitionCSVInput(RunningStage.TRAINING, train_file, **dataset_kwargs),
            SpeechRecognitionCSVInput(RunningStage.VALIDATING, val_file, **dataset_kwargs),
            SpeechRecognitionCSVInput(RunningStage.TESTING, test_file, **dataset_kwargs),
            SpeechRecognitionCSVInput(RunningStage.PREDICTING, predict_file, **dataset_kwargs),
            input_transform=cls.input_transform_cls(
                train_transform, val_transform, test_transform, predict_transform, sampling_rate
            ),
            **data_module_kwargs,
        )

    @classmethod
    def from_json(
        cls,
        input_fields: Union[str, Sequence[str]],
        target_fields: Optional[str] = None,
        train_file: Optional[str] = None,
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        predict_file: Optional[str] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        field: Optional[str] = None,
        sampling_rate: int = 16000,
        **data_module_kwargs: Any,
    ) -> "SpeechRecognitionData":
        dataset_kwargs = dict(
            input_key=input_fields,
            target_key=target_fields,
            sampling_rate=sampling_rate,
            field=field,
        )
        return cls(
            SpeechRecognitionJSONInput(RunningStage.TRAINING, train_file, **dataset_kwargs),
            SpeechRecognitionJSONInput(RunningStage.VALIDATING, val_file, **dataset_kwargs),
            SpeechRecognitionJSONInput(RunningStage.TESTING, test_file, **dataset_kwargs),
            SpeechRecognitionJSONInput(RunningStage.PREDICTING, predict_file, **dataset_kwargs),
            input_transform=cls.input_transform_cls(
                train_transform, val_transform, test_transform, predict_transform, sampling_rate
            ),
            **data_module_kwargs,
        )

    @classmethod
    def from_datasets(
        cls,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        predict_dataset: Optional[Dataset] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        sampling_rate: int = 16000,
        **data_module_kwargs,
    ) -> "SpeechRecognitionData":
        return cls(
            SpeechRecognitionDatasetInput(RunningStage.TRAINING, train_dataset, sampling_rate=sampling_rate),
            SpeechRecognitionDatasetInput(RunningStage.VALIDATING, val_dataset, sampling_rate=sampling_rate),
            SpeechRecognitionDatasetInput(RunningStage.TESTING, test_dataset, sampling_rate=sampling_rate),
            SpeechRecognitionDatasetInput(RunningStage.PREDICTING, predict_dataset, sampling_rate=sampling_rate),
            input_transform=cls.input_transform_cls(
                train_transform, val_transform, test_transform, predict_transform, sampling_rate
            ),
            **data_module_kwargs,
        )
