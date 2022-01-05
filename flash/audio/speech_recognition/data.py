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
from typing import Any, Dict, Optional, Sequence, Type, Union

from torch.utils.data import Dataset

from flash.audio.speech_recognition.input import (
    SpeechRecognitionCSVInput,
    SpeechRecognitionDatasetInput,
    SpeechRecognitionJSONInput,
    SpeechRecognitionPathsInput,
)
from flash.audio.speech_recognition.output_transform import SpeechRecognitionOutputTransform
from flash.core.data.data_module import DataModule
from flash.core.data.data_pipeline import DataPipelineState
from flash.core.data.io.input import Input
from flash.core.data.io.input_transform import INPUT_TRANSFORM_TYPE, InputTransform
from flash.core.registry import FlashRegistry
from flash.core.utilities.stages import RunningStage


class SpeechRecognitionData(DataModule):
    """Data Module for text classification tasks."""

    input_transform_cls = InputTransform
    output_transform_cls = SpeechRecognitionOutputTransform
    input_transforms_registry = FlashRegistry("input_transforms")

    @classmethod
    def from_files(
        cls,
        train_files: Optional[Sequence[str]] = None,
        train_targets: Optional[Sequence[str]] = None,
        val_files: Optional[Sequence[str]] = None,
        val_targets: Optional[Sequence[str]] = None,
        test_files: Optional[Sequence[str]] = None,
        test_targets: Optional[Sequence[str]] = None,
        predict_files: Optional[Sequence[str]] = None,
        sampling_rate: int = 16000,
        train_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        input_cls: Type[Input] = SpeechRecognitionPathsInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "SpeechRecognitionData":
        """Load the :class:`~flash.audio.speech_recognition.data.SpeechRecognitionData` from lists of audio files
        and corresponding lists of targets.

        The supported file extensions are: ``wav``, ``ogg``, ``flac``, ``mat``, and ``mp3``.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_files: The list of audio files to use when training.
            train_targets: The list of targets (ground truth speech transcripts) to use when training.
            val_files: The list of audio files to use when validating.
            val_targets: The list of targets (ground truth speech transcripts) to use when validating.
            test_files: The list of audio files to use when testing.
            test_targets: The list of targets (ground truth speech transcripts) to use when testing.
            predict_files: The list of audio files to use when predicting.
            sampling_rate: Sampling rate to use when loading the audio files.
            train_transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use when training.
            val_transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use when validating.
            test_transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use when testing.
            predict_transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use when
              predicting.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
              :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.audio.speech_recognition.data.SpeechRecognitionData`.

        Examples
        ________

        .. testsetup::

            >>> import numpy as np
            >>> import soundfile as sf
            >>> samplerate = 44100
            >>> data = np.random.uniform(-1, 1, size=(samplerate * 3, 2))
            >>> _ = [sf.write(f"speech_{i}.wav", data, samplerate, subtype='PCM_24') for i in range(1, 4)]
            >>> _ = [sf.write(f"predict_speech_{i}.wav", data, samplerate, subtype='PCM_24') for i in range(1, 4)]

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.audio import SpeechRecognitionData, SpeechRecognition
            >>> datamodule = SpeechRecognitionData.from_files(
            ...     train_files=["speech_1.wav", "speech_2.wav", "speech_3.wav"],
            ...     train_targets=["some speech", "some other speech", "some more speech"],
            ...     predict_files=["predict_speech_1.wav", "predict_speech_2.wav", "predict_speech_3.wav"],
            ...     batch_size=2,
            ... )
            >>> model = SpeechRecognition(backbone="patrickvonplaten/wav2vec2_tiny_random_robust")
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> import os
            >>> _ = [os.remove(f"speech_{i}.wav") for i in range(1, 4)]
            >>> _ = [os.remove(f"predict_speech_{i}.wav") for i in range(1, 4)]
        """

        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
            sampling_rate=sampling_rate,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_files, train_targets, transform=train_transform, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_files, val_targets, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_files, test_targets, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_files, transform=predict_transform, **ds_kw),
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
        sampling_rate: int = 16000,
        train_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        input_cls: Type[Input] = SpeechRecognitionCSVInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "SpeechRecognitionData":

        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
            input_key=input_fields,
            target_key=target_fields,
            sampling_rate=sampling_rate,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_file, transform=train_transform, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_file, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_file, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_file, transform=predict_transform, **ds_kw),
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
        sampling_rate: int = 16000,
        field: Optional[str] = None,
        train_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        input_cls: Type[Input] = SpeechRecognitionJSONInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "SpeechRecognitionData":

        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
            input_key=input_fields,
            target_key=target_fields,
            sampling_rate=sampling_rate,
            field=field,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_file, transform=train_transform, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_file, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_file, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_file, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )

    @classmethod
    def from_datasets(
        cls,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        predict_dataset: Optional[Dataset] = None,
        train_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = InputTransform,
        sampling_rate: int = 16000,
        input_cls: Type[Input] = SpeechRecognitionDatasetInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "SpeechRecognitionData":

        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
            sampling_rate=sampling_rate,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_dataset, transform=train_transform, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_dataset, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_dataset, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_dataset, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )
