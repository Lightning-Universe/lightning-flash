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
from typing import Any, Callable, Collection, Dict, List, Optional, Sequence, Type, Union

import numpy as np
import pandas as pd
from torch import Tensor

from flash.audio.classification.input import (
    AudioClassificationCSVInput,
    AudioClassificationDataFrameInput,
    AudioClassificationFilesInput,
    AudioClassificationFolderInput,
    AudioClassificationNumpyInput,
    AudioClassificationTensorInput,
)
from flash.audio.classification.input_transform import AudioClassificationInputTransform
from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_module import DataModule
from flash.core.data.io.input import Input
from flash.core.data.io.input_transform import INPUT_TRANSFORM_TYPE
from flash.core.data.utilities.classification import TargetFormatter
from flash.core.data.utilities.paths import PATH_TYPE
from flash.core.utilities.imports import _AUDIO_TESTING
from flash.core.utilities.stages import RunningStage
from flash.image.classification.data import MatplotlibVisualization

# Skip doctests if requirements aren't available
if not _AUDIO_TESTING:
    __doctest_skip__ = ["AudioClassificationData", "AudioClassificationData.*"]


class AudioClassificationData(DataModule):
    """The ``AudioClassificationData`` class is a :class:`~flash.core.data.data_module.DataModule` with a set of
    classmethods for loading data for audio classification."""

    input_transform_cls = AudioClassificationInputTransform

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
        sampling_rate: int = 16000,
        n_fft: int = 400,
        input_cls: Type[Input] = AudioClassificationFilesInput,
        transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        target_formatter: Optional[TargetFormatter] = None,
        **data_module_kwargs: Any,
    ) -> "AudioClassificationData":
        """Load the :class:`~flash.audio.classification.data.AudioClassificationData` from lists of files and
        corresponding lists of targets.

        The supported file extensions for precomputed spectrograms are: ``.jpg``, ``.jpeg``, ``.png``, ``.ppm``,
        ``.bmp``, ``.pgm``, ``.tif``, ``.tiff``, ``.webp``, and ``.npy``.
        The supported file extensions for raw audio (where spectrograms will be computed automatically) are: ``.aiff``,
        ``.au``, ``.avr``, ``.caf``, ``.flac``, ``.mat``, ``.mat4``, ``.mat5``, ``.mpc2k``, ``.ogg``, ``.paf``,
        ``.pvf``, ``.rf64``, ``.ircam``, ``.voc``, ``.w64``, ``.wav``, ``.nist``, and ``.wavex``.
        The targets can be in any of our
        :ref:`supported classification target formats <formatting_classification_targets>`.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_files: The list of spectrogram image files to use when training.
            train_targets: The list of targets to use when training.
            val_files: The list of spectrogram image files to use when validating.
            val_targets: The list of targets to use when validating.
            test_files: The list of spectrogram image files to use when testing.
            test_targets: The list of targets to use when testing.
            predict_files: The list of spectrogram image files to use when predicting.
            sampling_rate: Sampling rate to use when loading raw audio files.
            n_fft: The size of the FFT to use when creating spectrograms from raw audio.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.audio.classification.data.AudioClassificationData`.

        Examples
        ________

        .. testsetup::

            >>> from PIL import Image
            >>> rand_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))
            >>> _ = [rand_image.save(f"spectrogram_{i}.png") for i in range(1, 4)]
            >>> _ = [rand_image.save(f"predict_spectrogram_{i}.png") for i in range(1, 4)]

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.audio import AudioClassificationData
            >>> from flash.image import ImageClassifier
            >>> datamodule = AudioClassificationData.from_files(
            ...     train_files=["spectrogram_1.png", "spectrogram_2.png", "spectrogram_3.png"],
            ...     train_targets=["meow", "bark", "meow"],
            ...     predict_files=[
            ...         "predict_spectrogram_1.png",
            ...         "predict_spectrogram_2.png",
            ...         "predict_spectrogram_3.png",
            ...     ],
            ...     transform_kwargs=dict(spectrogram_size=(128, 128)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['bark', 'meow']
            >>> model = ImageClassifier(backbone="resnet18", num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> import os
            >>> _ = [os.remove(f"spectrogram_{i}.png") for i in range(1, 4)]
            >>> _ = [os.remove(f"predict_spectrogram_{i}.png") for i in range(1, 4)]
        """

        ds_kw = dict(
            sampling_rate=sampling_rate,
            n_fft=n_fft,
            target_formatter=target_formatter,
        )

        train_input = input_cls(RunningStage.TRAINING, train_files, train_targets, **ds_kw)
        ds_kw["target_formatter"] = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_files, val_targets, **ds_kw),
            input_cls(RunningStage.TESTING, test_files, test_targets, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_files, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[str] = None,
        val_folder: Optional[str] = None,
        test_folder: Optional[str] = None,
        predict_folder: Optional[str] = None,
        sampling_rate: int = 16000,
        n_fft: int = 400,
        input_cls: Type[Input] = AudioClassificationFolderInput,
        transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        target_formatter: Optional[TargetFormatter] = None,
        **data_module_kwargs: Any,
    ) -> "AudioClassificationData":
        """Load the :class:`~flash.audio.classification.data.AudioClassificationData` from folders containing
        spectrogram images.

        The supported file extensions for precomputed spectrograms are: ``.jpg``, ``.jpeg``, ``.png``, ``.ppm``,
        ``.bmp``, ``.pgm``, ``.tif``, ``.tiff``, ``.webp``, and ``.npy``.
        The supported file extensions for raw audio (where spectrograms will be computed automatically) are: ``.aiff``,
        ``.au``, ``.avr``, ``.caf``, ``.flac``, ``.mat``, ``.mat4``, ``.mat5``, ``.mpc2k``, ``.ogg``, ``.paf``,
        ``.pvf``, ``.rf64``, ``.ircam``, ``.voc``, ``.w64``, ``.wav``, ``.nist``, and ``.wavex``.
        For train, test, and validation data, the folders are expected to contain a sub-folder for each class.
        Here's the required structure:

        .. code-block::

            train_folder
            ├── meow
            │   ├── spectrogram_1.png
            │   ├── spectrogram_3.png
            │   ...
            └── bark
                ├── spectrogram_2.png
                ...

        For prediction, the folder is expected to contain the files for inference, like this:

        .. code-block::

            predict_folder
            ├── predict_spectrogram_1.png
            ├── predict_spectrogram_2.png
            ├── predict_spectrogram_3.png
            ...

        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_folder: The folder containing spectrogram images to use when training.
            val_folder: The folder containing spectrogram images to use when validating.
            test_folder: The folder containing spectrogram images to use when testing.
            predict_folder: The folder containing spectrogram images to use when predicting.
            sampling_rate: Sampling rate to use when loading raw audio files.
            n_fft: The size of the FFT to use when creating spectrograms from raw audio.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.image.classification.data.ImageClassificationData`.

        Examples
        ________

        .. testsetup::

            >>> import os
            >>> from PIL import Image
            >>> rand_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))
            >>> os.makedirs(os.path.join("train_folder", "meow"), exist_ok=True)
            >>> os.makedirs(os.path.join("train_folder", "bark"), exist_ok=True)
            >>> os.makedirs("predict_folder", exist_ok=True)
            >>> rand_image.save(os.path.join("train_folder", "meow", "spectrogram_1.png"))
            >>> rand_image.save(os.path.join("train_folder", "bark", "spectrogram_2.png"))
            >>> rand_image.save(os.path.join("train_folder", "meow", "spectrogram_3.png"))
            >>> _ = [rand_image.save(
            ...     os.path.join("predict_folder", f"predict_spectrogram_{i}.png")
            ... ) for i in range(1, 4)]

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.audio import AudioClassificationData
            >>> from flash.image import ImageClassifier
            >>> datamodule = AudioClassificationData.from_folders(
            ...     train_folder="train_folder",
            ...     predict_folder="predict_folder",
            ...     transform_kwargs=dict(spectrogram_size=(128, 128)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['bark', 'meow']
            >>> model = ImageClassifier(backbone="resnet18", num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> import shutil
            >>> shutil.rmtree("train_folder")
            >>> shutil.rmtree("predict_folder")
        """

        ds_kw = dict(
            sampling_rate=sampling_rate,
            n_fft=n_fft,
            target_formatter=target_formatter,
        )

        train_input = input_cls(RunningStage.TRAINING, train_folder, **ds_kw)
        ds_kw["target_formatter"] = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_folder, **ds_kw),
            input_cls(RunningStage.TESTING, test_folder, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_folder, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_numpy(
        cls,
        train_data: Optional[Collection[np.ndarray]] = None,
        train_targets: Optional[Collection[Any]] = None,
        val_data: Optional[Collection[np.ndarray]] = None,
        val_targets: Optional[Sequence[Any]] = None,
        test_data: Optional[Collection[np.ndarray]] = None,
        test_targets: Optional[Sequence[Any]] = None,
        predict_data: Optional[Collection[np.ndarray]] = None,
        input_cls: Type[Input] = AudioClassificationNumpyInput,
        transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        target_formatter: Optional[TargetFormatter] = None,
        **data_module_kwargs: Any,
    ) -> "AudioClassificationData":
        """Load the :class:`~flash.audio.classification.data.AudioClassificationData` from numpy arrays (or lists
        of arrays) and corresponding lists of targets.

        The targets can be in any of our
        :ref:`supported classification target formats <formatting_classification_targets>`.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_data: The numpy array or list of arrays to use when training.
            train_targets: The list of targets to use when training.
            val_data: The numpy array or list of arrays to use when validating.
            val_targets: The list of targets to use when validating.
            test_data: The numpy array or list of arrays to use when testing.
            test_targets: The list of targets to use when testing.
            predict_data: The numpy array or list of arrays to use when predicting.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.audio.classification.data.AudioClassificationData`.

        Examples
        ________

        .. doctest::

            >>> import numpy as np
            >>> from flash import Trainer
            >>> from flash.audio import AudioClassificationData
            >>> from flash.image import ImageClassifier
            >>> datamodule = AudioClassificationData.from_numpy(
            ...     train_data=[np.random.rand(3, 64, 64), np.random.rand(3, 64, 64), np.random.rand(3, 64, 64)],
            ...     train_targets=["meow", "bark", "meow"],
            ...     predict_data=[np.random.rand(3, 64, 64)],
            ...     transform_kwargs=dict(spectrogram_size=(128, 128)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['bark', 'meow']
            >>> model = ImageClassifier(backbone="resnet18", num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...
        """

        ds_kw = dict(
            target_formatter=target_formatter,
        )

        train_input = input_cls(RunningStage.TRAINING, train_data, train_targets, **ds_kw)
        ds_kw["target_formatter"] = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_data, val_targets, **ds_kw),
            input_cls(RunningStage.TESTING, test_data, test_targets, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_data, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_tensors(
        cls,
        train_data: Optional[Collection[Tensor]] = None,
        train_targets: Optional[Collection[Any]] = None,
        val_data: Optional[Collection[Tensor]] = None,
        val_targets: Optional[Sequence[Any]] = None,
        test_data: Optional[Collection[Tensor]] = None,
        test_targets: Optional[Sequence[Any]] = None,
        predict_data: Optional[Collection[Tensor]] = None,
        input_cls: Type[Input] = AudioClassificationTensorInput,
        transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        target_formatter: Optional[TargetFormatter] = None,
        **data_module_kwargs: Any,
    ) -> "AudioClassificationData":
        """Load the :class:`~flash.audio.classification.data.AudioClassificationData` from torch tensors (or lists
        of tensors) and corresponding lists of targets.

        The targets can be in any of our
        :ref:`supported classification target formats <formatting_classification_targets>`.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_data: The torch tensor or list of tensors to use when training.
            train_targets: The list of targets to use when training.
            val_data: The torch tensor or list of tensors to use when validating.
            val_targets: The list of targets to use when validating.
            test_data: The torch tensor or list of tensors to use when testing.
            test_targets: The list of targets to use when testing.
            predict_data: The torch tensor or list of tensors to use when predicting.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.audio.classification.data.AudioClassificationData`.

        Examples
        ________

        .. doctest::

            >>> import torch
            >>> from flash import Trainer
            >>> from flash.audio import AudioClassificationData
            >>> from flash.image import ImageClassifier
            >>> datamodule = AudioClassificationData.from_tensors(
            ...     train_data=[torch.rand(3, 64, 64), torch.rand(3, 64, 64), torch.rand(3, 64, 64)],
            ...     train_targets=["meow", "bark", "meow"],
            ...     predict_data=[torch.rand(3, 64, 64)],
            ...     transform_kwargs=dict(spectrogram_size=(128, 128)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['bark', 'meow']
            >>> model = ImageClassifier(backbone="resnet18", num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...
        """

        ds_kw = dict(
            target_formatter=target_formatter,
        )

        train_input = input_cls(RunningStage.TRAINING, train_data, train_targets, **ds_kw)
        ds_kw["target_formatter"] = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_data, val_targets, **ds_kw),
            input_cls(RunningStage.TESTING, test_data, test_targets, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_data, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_data_frame(
        cls,
        input_field: str,
        target_fields: Optional[Union[str, Sequence[str]]] = None,
        train_data_frame: Optional[pd.DataFrame] = None,
        train_images_root: Optional[str] = None,
        train_resolver: Optional[Callable[[str, str], str]] = None,
        val_data_frame: Optional[pd.DataFrame] = None,
        val_images_root: Optional[str] = None,
        val_resolver: Optional[Callable[[str, str], str]] = None,
        test_data_frame: Optional[pd.DataFrame] = None,
        test_images_root: Optional[str] = None,
        test_resolver: Optional[Callable[[str, str], str]] = None,
        predict_data_frame: Optional[pd.DataFrame] = None,
        predict_images_root: Optional[str] = None,
        predict_resolver: Optional[Callable[[str, str], str]] = None,
        sampling_rate: int = 16000,
        n_fft: int = 400,
        input_cls: Type[Input] = AudioClassificationDataFrameInput,
        transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        target_formatter: Optional[TargetFormatter] = None,
        **data_module_kwargs: Any,
    ) -> "AudioClassificationData":
        """Load the :class:`~flash.audio.classification.data.AudioClassificationData` from pandas DataFrame objects
        containing spectrogram image file paths and their corresponding targets.

        Input spectrogram image paths will be extracted from the ``input_field`` in the DataFrame.
        The supported file extensions for precomputed spectrograms are: ``.jpg``, ``.jpeg``, ``.png``, ``.ppm``,
        ``.bmp``, ``.pgm``, ``.tif``, ``.tiff``, ``.webp``, and ``.npy``.
        The supported file extensions for raw audio (where spectrograms will be computed automatically) are: ``.aiff``,
        ``.au``, ``.avr``, ``.caf``, ``.flac``, ``.mat``, ``.mat4``, ``.mat5``, ``.mpc2k``, ``.ogg``, ``.paf``,
        ``.pvf``, ``.rf64``, ``.ircam``, ``.voc``, ``.w64``, ``.wav``, ``.nist``, and ``.wavex``.
        The targets will be extracted from the ``target_fields`` in the DataFrame and can be in any of our
        :ref:`supported classification target formats <formatting_classification_targets>`.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            input_field: The field (column name) in the DataFrames containing the spectrogram image file paths.
            target_fields: The field (column name) or list of fields in the DataFrames containing the targets.
            train_data_frame: The pandas DataFrame to use when training.
            train_images_root: The root directory containing train spectrogram images.
            train_resolver: Optionally provide a function which converts an entry from the ``input_field`` into a
                spectrogram image file path.
            val_data_frame: The pandas DataFrame to use when validating.
            val_images_root: The root directory containing validation spectrogram images.
            val_resolver: Optionally provide a function which converts an entry from the ``input_field`` into a
                spectrogram image file path.
            test_data_frame: The pandas DataFrame to use when testing.
            test_images_root: The root directory containing test spectrogram images.
            test_resolver: Optionally provide a function which converts an entry from the ``input_field`` into a
                spectrogram image file path.
            predict_data_frame: The pandas DataFrame to use when predicting.
            predict_images_root: The root directory containing predict spectrogram images.
            predict_resolver: Optionally provide a function which converts an entry from the ``input_field`` into a
                spectrogram image file path.
            sampling_rate: Sampling rate to use when loading raw audio files.
            n_fft: The size of the FFT to use when creating spectrograms from raw audio.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.image.classification.data.ImageClassificationData`.

        Examples
        ________

        .. testsetup::

            >>> import os
            >>> from PIL import Image
            >>> rand_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))
            >>> os.makedirs("train_folder", exist_ok=True)
            >>> os.makedirs("predict_folder", exist_ok=True)
            >>> _ = [rand_image.save(os.path.join("train_folder", f"spectrogram_{i}.png")) for i in range(1, 4)]
            >>> _ = [rand_image.save(
            ...     os.path.join("predict_folder", f"predict_spectrogram_{i}.png")
            ... ) for i in range(1, 4)]

        .. doctest::

            >>> from pandas import DataFrame
            >>> from flash import Trainer
            >>> from flash.audio import AudioClassificationData
            >>> from flash.image import ImageClassifier
            >>> train_data_frame = DataFrame.from_dict(
            ...     {
            ...         "images": ["spectrogram_1.png", "spectrogram_2.png", "spectrogram_3.png"],
            ...         "targets": ["meow", "bark", "meow"],
            ...     }
            ... )
            >>> predict_data_frame = DataFrame.from_dict(
            ...     {
            ...         "images": [
            ...             "predict_spectrogram_1.png",
            ...             "predict_spectrogram_2.png",
            ...             "predict_spectrogram_3.png",
            ...         ],
            ...     }
            ... )
            >>> datamodule = AudioClassificationData.from_data_frame(
            ...     "images",
            ...     "targets",
            ...     train_data_frame=train_data_frame,
            ...     train_images_root="train_folder",
            ...     predict_data_frame=predict_data_frame,
            ...     predict_images_root="predict_folder",
            ...     transform_kwargs=dict(spectrogram_size=(128, 128)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['bark', 'meow']
            >>> model = ImageClassifier(backbone="resnet18", num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> import shutil
            >>> shutil.rmtree("train_folder")
            >>> shutil.rmtree("predict_folder")
            >>> del train_data_frame
            >>> del predict_data_frame
        """

        ds_kw = dict(
            sampling_rate=sampling_rate,
            n_fft=n_fft,
            target_formatter=target_formatter,
        )

        train_data = (train_data_frame, input_field, target_fields, train_images_root, train_resolver)
        val_data = (val_data_frame, input_field, target_fields, val_images_root, val_resolver)
        test_data = (test_data_frame, input_field, target_fields, test_images_root, test_resolver)
        predict_data = (predict_data_frame, input_field, None, predict_images_root, predict_resolver)

        train_input = input_cls(RunningStage.TRAINING, *train_data, **ds_kw)
        ds_kw["target_formatter"] = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, *val_data, **ds_kw),
            input_cls(RunningStage.TESTING, *test_data, **ds_kw),
            input_cls(RunningStage.PREDICTING, *predict_data, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_csv(
        cls,
        input_field: str,
        target_fields: Optional[Union[str, List[str]]] = None,
        train_file: Optional[PATH_TYPE] = None,
        train_images_root: Optional[PATH_TYPE] = None,
        train_resolver: Optional[Callable[[PATH_TYPE, Any], PATH_TYPE]] = None,
        val_file: Optional[PATH_TYPE] = None,
        val_images_root: Optional[PATH_TYPE] = None,
        val_resolver: Optional[Callable[[PATH_TYPE, Any], PATH_TYPE]] = None,
        test_file: Optional[str] = None,
        test_images_root: Optional[str] = None,
        test_resolver: Optional[Callable[[PATH_TYPE, Any], PATH_TYPE]] = None,
        predict_file: Optional[str] = None,
        predict_images_root: Optional[str] = None,
        predict_resolver: Optional[Callable[[PATH_TYPE, Any], PATH_TYPE]] = None,
        sampling_rate: int = 16000,
        n_fft: int = 400,
        input_cls: Type[Input] = AudioClassificationCSVInput,
        transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        target_formatter: Optional[TargetFormatter] = None,
        **data_module_kwargs: Any,
    ) -> "AudioClassificationData":
        """Load the :class:`~flash.audio.classification.data.AudioClassificationData` from CSV files containing
        spectrogram image file paths and their corresponding targets.

        Input spectrogram images will be extracted from the ``input_field`` column in the CSV files.
        The supported file extensions for precomputed spectrograms are: ``.jpg``, ``.jpeg``, ``.png``, ``.ppm``,
        ``.bmp``, ``.pgm``, ``.tif``, ``.tiff``, ``.webp``, and ``.npy``.
        The supported file extensions for raw audio (where spectrograms will be computed automatically) are: ``.aiff``,
        ``.au``, ``.avr``, ``.caf``, ``.flac``, ``.mat``, ``.mat4``, ``.mat5``, ``.mpc2k``, ``.ogg``, ``.paf``,
        ``.pvf``, ``.rf64``, ``.ircam``, ``.voc``, ``.w64``, ``.wav``, ``.nist``, and ``.wavex``.
        The targets will be extracted from the ``target_fields`` in the CSV files and can be in any of our
        :ref:`supported classification target formats <formatting_classification_targets>`.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            input_field: The field (column name) in the CSV files containing the spectrogram image file paths.
            target_fields: The field (column name) or list of fields in the CSV files containing the targets.
            train_file: The CSV file to use when training.
            train_images_root: The root directory containing train spectrogram images.
            train_resolver: Optionally provide a function which converts an entry from the ``input_field`` into a
                spectrogram image file path.
            val_file: The CSV file to use when validating.
            val_images_root: The root directory containing validation spectrogram images.
            val_resolver: Optionally provide a function which converts an entry from the ``input_field`` into a
                spectrogram image file path.
            test_file: The CSV file to use when testing.
            test_images_root: The root directory containing test spectrogram images.
            test_resolver: Optionally provide a function which converts an entry from the ``input_field`` into a
                spectrogram image file path.
            predict_file: The CSV file to use when predicting.
            predict_images_root: The root directory containing predict spectrogram images.
            predict_resolver: Optionally provide a function which converts an entry from the ``input_field`` into a
                spectrogram image file path.
            sampling_rate: Sampling rate to use when loading raw audio files.
            n_fft: The size of the FFT to use when creating spectrograms from raw audio.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.audio.classification.data.AudioClassificationData`.

        Examples
        ________

        The files can be in Comma Separated Values (CSV) format with either a ``.csv`` or ``.txt`` extension.

        .. testsetup::

            >>> import os
            >>> from PIL import Image
            >>> from pandas import DataFrame
            >>> rand_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))
            >>> os.makedirs("train_folder", exist_ok=True)
            >>> os.makedirs("predict_folder", exist_ok=True)
            >>> _ = [rand_image.save(os.path.join("train_folder", f"spectrogram_{i}.png")) for i in range(1, 4)]
            >>> _ = [rand_image.save(
            ...     os.path.join("predict_folder", f"predict_spectrogram_{i}.png")
            ... ) for i in range(1, 4)]
            >>> DataFrame.from_dict({
            ...     "images": ["spectrogram_1.png", "spectrogram_2.png", "spectrogram_3.png"],
            ...     "targets": ["meow", "bark", "meow"],
            ... }).to_csv("train_data.csv", index=False)
            >>> DataFrame.from_dict({
            ...     "images": ["predict_spectrogram_1.png", "predict_spectrogram_2.png", "predict_spectrogram_3.png"],
            ... }).to_csv("predict_data.csv", index=False)

        The file ``train_data.csv`` contains the following:

        .. code-block::

            images,targets
            spectrogram_1.png,meow
            spectrogram_2.png,bark
            spectrogram_3.png,meow

        The file ``predict_data.csv`` contains the following:

        .. code-block::

            images
            predict_spectrogram_1.png
            predict_spectrogram_2.png
            predict_spectrogram_3.png

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.audio import AudioClassificationData
            >>> from flash.image import ImageClassifier
            >>> datamodule = AudioClassificationData.from_csv(
            ...     "images",
            ...     "targets",
            ...     train_file="train_data.csv",
            ...     train_images_root="train_folder",
            ...     predict_file="predict_data.csv",
            ...     predict_images_root="predict_folder",
            ...     transform_kwargs=dict(spectrogram_size=(128, 128)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['bark', 'meow']
            >>> model = ImageClassifier(backbone="resnet18", num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> import shutil
            >>> shutil.rmtree("train_folder")
            >>> shutil.rmtree("predict_folder")
            >>> os.remove("train_data.csv")
            >>> os.remove("predict_data.csv")

        Alternatively, the files can be in Tab Separated Values (TSV) format with a ``.tsv`` extension.

        .. testsetup::

            >>> import os
            >>> from PIL import Image
            >>> from pandas import DataFrame
            >>> rand_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype="uint8"))
            >>> os.makedirs("train_folder", exist_ok=True)
            >>> os.makedirs("predict_folder", exist_ok=True)
            >>> _ = [rand_image.save(os.path.join("train_folder", f"spectrogram_{i}.png")) for i in range(1, 4)]
            >>> _ = [rand_image.save(
            ...     os.path.join("predict_folder", f"predict_spectrogram_{i}.png")
            ... ) for i in range(1, 4)]
            >>> DataFrame.from_dict({
            ...     "images": ["spectrogram_1.png", "spectrogram_2.png", "spectrogram_3.png"],
            ...     "targets": ["meow", "bark", "meow"],
            ... }).to_csv("train_data.tsv", sep="\\t", index=False)
            >>> DataFrame.from_dict({
            ...     "images": ["predict_spectrogram_1.png", "predict_spectrogram_2.png", "predict_spectrogram_3.png"],
            ... }).to_csv("predict_data.tsv", sep="\\t", index=False)

        The file ``train_data.tsv`` contains the following:

        .. code-block::

            images              targets
            spectrogram_1.png   meow
            spectrogram_2.png   bark
            spectrogram_3.png   meow

        The file ``predict_data.tsv`` contains the following:

        .. code-block::

            images
            predict_spectrogram_1.png
            predict_spectrogram_2.png
            predict_spectrogram_3.png

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.audio import AudioClassificationData
            >>> from flash.image import ImageClassifier
            >>> datamodule = AudioClassificationData.from_csv(
            ...     "images",
            ...     "targets",
            ...     train_file="train_data.tsv",
            ...     train_images_root="train_folder",
            ...     predict_file="predict_data.tsv",
            ...     predict_images_root="predict_folder",
            ...     transform_kwargs=dict(spectrogram_size=(128, 128)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['bark', 'meow']
            >>> model = ImageClassifier(backbone="resnet18", num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> import shutil
            >>> shutil.rmtree("train_folder")
            >>> shutil.rmtree("predict_folder")
            >>> os.remove("train_data.tsv")
            >>> os.remove("predict_data.tsv")
        """

        ds_kw = dict(
            sampling_rate=sampling_rate,
            n_fft=n_fft,
            target_formatter=target_formatter,
        )

        train_data = (train_file, input_field, target_fields, train_images_root, train_resolver)
        val_data = (val_file, input_field, target_fields, val_images_root, val_resolver)
        test_data = (test_file, input_field, target_fields, test_images_root, test_resolver)
        predict_data = (predict_file, input_field, None, predict_images_root, predict_resolver)

        train_input = input_cls(RunningStage.TRAINING, *train_data, **ds_kw)
        ds_kw["target_formatter"] = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, *val_data, **ds_kw),
            input_cls(RunningStage.TESTING, *test_data, **ds_kw),
            input_cls(RunningStage.PREDICTING, *predict_data, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    def set_block_viz_window(self, value: bool) -> None:
        """Setter method to switch on/off matplotlib to pop up windows."""
        self.data_fetcher.block_viz_window = value

    @staticmethod
    def configure_data_fetcher(*args, **kwargs) -> BaseDataFetcher:
        return MatplotlibVisualization(*args, **kwargs)
