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

from flash.audio.speech_recognition.input import (
    SpeechRecognitionCSVInput,
    SpeechRecognitionDatasetInput,
    SpeechRecognitionJSONInput,
    SpeechRecognitionPathsInput,
)
from flash.audio.speech_recognition.output_transform import SpeechRecognitionOutputTransform
from flash.core.data.data_module import DataModule
from flash.core.data.io.input import Input
from flash.core.data.io.input_transform import INPUT_TRANSFORM_TYPE, InputTransform
from flash.core.utilities.imports import _AUDIO_TESTING
from flash.core.utilities.stages import RunningStage

# Skip doctests if requirements aren't available
if not _AUDIO_TESTING:
    __doctest_skip__ = ["SpeechRecognitionData", "SpeechRecognitionData.*"]


class SpeechRecognitionData(DataModule):
    """The ``SpeechRecognitionData`` class is a :class:`~flash.core.data.data_module.DataModule` with a set of
    classmethods for loading data for speech recognition."""

    input_transform_cls = InputTransform
    output_transform_cls = SpeechRecognitionOutputTransform

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
        input_cls: Type[Input] = SpeechRecognitionPathsInput,
        transform: INPUT_TRANSFORM_TYPE = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "SpeechRecognitionData":
        """Load the :class:`~flash.audio.speech_recognition.data.SpeechRecognitionData` from lists of audio files
        and corresponding lists of targets.

        The supported file extensions are: ``.aiff``, ``.au``, ``.avr``, ``.caf``, ``.flac``, ``.mat``, ``.mat4``,
        ``.mat5``, ``.mpc2k``, ``.ogg``, ``.paf``, ``.pvf``, ``.rf64``, ``.ircam``, ``.voc``, ``.w64``,
        ``.wav``, ``.nist``, and ``.wavex``.
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
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
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
            >>> samplerate = 1000
            >>> data = np.random.uniform(-1, 1, size=(samplerate, 2))
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
            sampling_rate=sampling_rate,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_files, train_targets, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_files, val_targets, **ds_kw),
            input_cls(RunningStage.TESTING, test_files, test_targets, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_files, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_csv(
        cls,
        input_field: str,
        target_field: Optional[str] = None,
        train_file: Optional[str] = None,
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        predict_file: Optional[str] = None,
        sampling_rate: int = 16000,
        input_cls: Type[Input] = SpeechRecognitionCSVInput,
        transform: INPUT_TRANSFORM_TYPE = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "SpeechRecognitionData":
        """Load the :class:`~flash.audio.speech_recognition.data.SpeechRecognitionData` from CSV files containing
        audio file paths and their corresponding targets.

        Input audio file paths will be extracted from the ``input_field`` column in the CSV files.
        The supported file extensions are: ``.aiff``, ``.au``, ``.avr``, ``.caf``, ``.flac``, ``.mat``, ``.mat4``,
        ``.mat5``, ``.mpc2k``, ``.ogg``, ``.paf``, ``.pvf``, ``.rf64``, ``.ircam``, ``.voc``, ``.w64``,
        ``.wav``, ``.nist``, and ``.wavex``.
        The targets will be extracted from the ``target_field`` in the CSV files.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            input_field: The field (column name) in the CSV files containing the audio file paths.
            target_field: The field (column name) in the CSV files containing the targets.
            train_file: The CSV file to use when training.
            val_file: The CSV file to use when validating.
            test_file: The CSV file to use when testing.
            predict_file: The CSV file to use when predicting.
            sampling_rate: Sampling rate to use when loading the audio files.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
              :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.audio.speech_recognition.data.SpeechRecognitionData`.

        Examples
        ________

        The files can be in Comma Separated Values (CSV) format with either a ``.csv`` or ``.txt`` extension.

        .. testsetup::

            >>> import numpy as np
            >>> from pandas import DataFrame
            >>> import soundfile as sf
            >>> samplerate = 1000
            >>> data = np.random.uniform(-1, 1, size=(samplerate, 2))
            >>> _ = [sf.write(f"speech_{i}.wav", data, samplerate, subtype='PCM_24') for i in range(1, 4)]
            >>> _ = [sf.write(f"predict_speech_{i}.wav", data, samplerate, subtype='PCM_24') for i in range(1, 4)]
            >>> DataFrame.from_dict({
            ...     "speech_files": ["speech_1.wav", "speech_2.wav", "speech_3.wav"],
            ...     "targets": ["some speech", "some other speech", "some more speech"],
            ... }).to_csv("train_data.csv", index=False)
            >>> DataFrame.from_dict({
            ...     "speech_files": ["predict_speech_1.wav", "predict_speech_2.wav", "predict_speech_3.wav"],
            ... }).to_csv("predict_data.csv", index=False)

        The file ``train_data.csv`` contains the following:

        .. code-block::

            speech_files,targets
            speech_1.wav,some speech
            speech_2.wav,some other speech
            speech_3.wav,some more speech

        The file ``predict_data.csv`` contains the following:

        .. code-block::

            speech_files
            predict_speech_1.wav
            predict_speech_2.wav
            predict_speech_3.wav

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.audio import SpeechRecognitionData, SpeechRecognition
            >>> datamodule = SpeechRecognitionData.from_csv(
            ...     "speech_files",
            ...     "targets",
            ...     train_file="train_data.csv",
            ...     predict_file="predict_data.csv",
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
            >>> os.remove("train_data.csv")
            >>> os.remove("predict_data.csv")

        Alternatively, the files can be in Tab Separated Values (TSV) format with either a ``.tsv``.

        .. testsetup::

            >>> import numpy as np
            >>> from pandas import DataFrame
            >>> import soundfile as sf
            >>> samplerate = 1000
            >>> data = np.random.uniform(-1, 1, size=(samplerate, 2))
            >>> _ = [sf.write(f"speech_{i}.wav", data, samplerate, subtype='PCM_24') for i in range(1, 4)]
            >>> _ = [sf.write(f"predict_speech_{i}.wav", data, samplerate, subtype='PCM_24') for i in range(1, 4)]
            >>> DataFrame.from_dict({
            ...     "speech_files": ["speech_1.wav", "speech_2.wav", "speech_3.wav"],
            ...     "targets": ["some speech", "some other speech", "some more speech"],
            ... }).to_csv("train_data.tsv", sep="\\t", index=False)
            >>> DataFrame.from_dict({
            ...     "speech_files": ["predict_speech_1.wav", "predict_speech_2.wav", "predict_speech_3.wav"],
            ... }).to_csv("predict_data.tsv", sep="\\t", index=False)

        The file ``train_data.tsv`` contains the following:

        .. code-block::

            speech_files    targets
            speech_1.wav    some speech
            speech_2.wav    some other speech
            speech_3.wav    some more speech

        The file ``predict_data.tsv`` contains the following:

        .. code-block::

            speech_files
            predict_speech_1.wav
            predict_speech_2.wav
            predict_speech_3.wav

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.audio import SpeechRecognitionData, SpeechRecognition
            >>> datamodule = SpeechRecognitionData.from_csv(
            ...     "speech_files",
            ...     "targets",
            ...     train_file="train_data.tsv",
            ...     predict_file="predict_data.tsv",
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
            >>> os.remove("train_data.tsv")
            >>> os.remove("predict_data.tsv")
        """

        ds_kw = dict(
            input_key=input_field,
            sampling_rate=sampling_rate,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_file, target_key=target_field, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_file, target_key=target_field, **ds_kw),
            input_cls(RunningStage.TESTING, test_file, target_key=target_field, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_file, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_json(
        cls,
        input_field: str,
        target_field: Optional[str] = None,
        train_file: Optional[str] = None,
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        predict_file: Optional[str] = None,
        sampling_rate: int = 16000,
        field: Optional[str] = None,
        input_cls: Type[Input] = SpeechRecognitionJSONInput,
        transform: INPUT_TRANSFORM_TYPE = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "SpeechRecognitionData":
        """Load the :class:`~flash.audio.speech_recognition.data.SpeechRecognitionData` from JSON files containing
        audio file paths and their corresponding targets.

        Input audio file paths will be extracted from the ``input_field`` field in the JSON files.
        The supported file extensions are: ``.aiff``, ``.au``, ``.avr``, ``.caf``, ``.flac``, ``.mat``, ``.mat4``,
        ``.mat5``, ``.mpc2k``, ``.ogg``, ``.paf``, ``.pvf``, ``.rf64``, ``.ircam``, ``.voc``, ``.w64``,
        ``.wav``, ``.nist``, and ``.wavex``.
        The targets will be extracted from the ``target_field`` field in the JSON files.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            input_field: The field in the JSON files containing the audio file paths.
            target_field: The field in the JSON files containing the targets.
            train_file: The JSON file to use when training.
            val_file: The JSON file to use when validating.
            test_file: The JSON file to use when testing.
            predict_file: The JSON file to use when predicting.
            sampling_rate: Sampling rate to use when loading the audio files.
            field: The field that holds the data in the JSON file.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
              :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.audio.speech_recognition.data.SpeechRecognitionData`.

        Examples
        ________

        .. testsetup::

            >>> import numpy as np
            >>> from pandas import DataFrame
            >>> import soundfile as sf
            >>> samplerate = 1000
            >>> data = np.random.uniform(-1, 1, size=(samplerate, 2))
            >>> _ = [sf.write(f"speech_{i}.wav", data, samplerate, subtype='PCM_24') for i in range(1, 4)]
            >>> _ = [sf.write(f"predict_speech_{i}.wav", data, samplerate, subtype='PCM_24') for i in range(1, 4)]
            >>> DataFrame.from_dict({
            ...     "speech_files": ["speech_1.wav", "speech_2.wav", "speech_3.wav"],
            ...     "targets": ["some speech", "some other speech", "some more speech"],
            ... }).to_json("train_data.json", orient="records", lines=True)
            >>> DataFrame.from_dict({
            ...     "speech_files": ["predict_speech_1.wav", "predict_speech_2.wav", "predict_speech_3.wav"],
            ... }).to_json("predict_data.json", orient="records", lines=True)

        The file ``train_data.json`` contains the following:

        .. code-block::

            {"speech_files":"speech_1.wav","targets":"some speech"}
            {"speech_files":"speech_2.wav","targets":"some other speech"}
            {"speech_files":"speech_3.wav","targets":"some more speech"}

        The file ``predict_data.json`` contains the following:

        .. code-block::

            {"speech_files":"predict_speech_1.wav"}
            {"speech_files":"predict_speech_2.wav"}
            {"speech_files":"predict_speech_3.wav"}

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.audio import SpeechRecognitionData, SpeechRecognition
            >>> datamodule = SpeechRecognitionData.from_json(
            ...     "speech_files",
            ...     "targets",
            ...     train_file="train_data.json",
            ...     predict_file="predict_data.json",
            ...     batch_size=2,
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Downloading...
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
            >>> os.remove("train_data.json")
            >>> os.remove("predict_data.json")
        """

        ds_kw = dict(
            input_key=input_field,
            sampling_rate=sampling_rate,
            field=field,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_file, target_key=target_field, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_file, target_key=target_field, **ds_kw),
            input_cls(RunningStage.TESTING, test_file, target_key=target_field, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_file, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_datasets(
        cls,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        predict_dataset: Optional[Dataset] = None,
        sampling_rate: int = 16000,
        input_cls: Type[Input] = SpeechRecognitionDatasetInput,
        transform: INPUT_TRANSFORM_TYPE = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "SpeechRecognitionData":
        """Load the :class:`~flash.audio.speech_recognition.data.SpeechRecognitionData` from PyTorch Dataset
        objects.

        The Dataset objects should be one of the following:

        * A PyTorch Dataset where the ``__getitem__`` returns a tuple: ``(file_path or , target)``
        * A PyTorch Dataset where the ``__getitem__`` returns a dict: ``{"input": file_path, "target": target}``

        The supported file extensions are: ``.aiff``, ``.au``, ``.avr``, ``.caf``, ``.flac``, ``.mat``, ``.mat4``,
        ``.mat5``, ``.mpc2k``, ``.ogg``, ``.paf``, ``.pvf``, ``.rf64``, ``.ircam``, ``.voc``, ``.w64``,
        ``.wav``, ``.nist``, and ``.wavex``.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_dataset: The Dataset to use when training.
            val_dataset: The Dataset to use when validating.
            test_dataset: The Dataset to use when testing.
            predict_dataset: The Dataset to use when predicting.
            sampling_rate: Sampling rate to use when loading the audio files.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
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
            >>> samplerate = 1000
            >>> data = np.random.uniform(-1, 1, size=(samplerate, 2))
            >>> _ = [sf.write(f"speech_{i}.wav", data, samplerate, subtype='PCM_24') for i in range(1, 4)]
            >>> _ = [sf.write(f"predict_speech_{i}.wav", data, samplerate, subtype='PCM_24') for i in range(1, 4)]

        A PyTorch Dataset where the ``__getitem__`` returns a tuple: ``(file_path, target)``:

        .. doctest::

            >>> from torch.utils.data import Dataset
            >>> from flash import Trainer
            >>> from flash.audio import SpeechRecognitionData, SpeechRecognition
            >>>
            >>> class CustomDataset(Dataset):
            ...     def __init__(self, files, targets=None):
            ...         self.files = files
            ...         self.targets = targets
            ...     def __getitem__(self, index):
            ...         if self.targets is not None:
            ...             return self.files[index], self.targets[index]
            ...         return self.files[index]
            ...     def __len__(self):
            ...         return len(self.files)
            ...
            >>>
            >>> datamodule = SpeechRecognitionData.from_datasets(
            ...     train_dataset=CustomDataset(
            ...         ["speech_1.wav", "speech_2.wav", "speech_3.wav"],
            ...         ["some speech", "some other speech", "some more speech"],
            ...     ),
            ...     predict_dataset=CustomDataset(
            ...         ["predict_speech_1.wav", "predict_speech_2.wav", "predict_speech_3.wav"],
            ...     ),
            ...     batch_size=2,
            ... )
            >>> model = SpeechRecognition(backbone="patrickvonplaten/wav2vec2_tiny_random_robust")
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        A PyTorch Dataset where the ``__getitem__`` returns a dict: ``{"input": file_path, "target": target}``:

        .. doctest::

            >>> from torch.utils.data import Dataset
            >>> from flash import Trainer
            >>> from flash.audio import SpeechRecognitionData, SpeechRecognition
            >>>
            >>> class CustomDataset(Dataset):
            ...     def __init__(self, files, targets=None):
            ...         self.files = files
            ...         self.targets = targets
            ...     def __getitem__(self, index):
            ...         if self.targets is not None:
            ...             return {"input": self.files[index], "target": self.targets[index]}
            ...         return {"input": self.files[index]}
            ...     def __len__(self):
            ...         return len(self.files)
            ...
            >>>
            >>> datamodule = SpeechRecognitionData.from_datasets(
            ...     train_dataset=CustomDataset(
            ...         ["speech_1.wav", "speech_2.wav", "speech_3.wav"],
            ...         ["some speech", "some other speech", "some more speech"],
            ...     ),
            ...     predict_dataset=CustomDataset(
            ...         ["predict_speech_1.wav", "predict_speech_2.wav", "predict_speech_3.wav"],
            ...     ),
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
            sampling_rate=sampling_rate,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_dataset, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_dataset, **ds_kw),
            input_cls(RunningStage.TESTING, test_dataset, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_dataset, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )
