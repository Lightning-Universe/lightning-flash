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

import numpy as np
import pandas as pd

from flash.core.data.io.classification_input import ClassificationInputMixin
from flash.core.data.io.input import DataKeys, Input
from flash.core.data.utilities.classification import MultiBinaryTargetFormatter, TargetFormatter
from flash.core.data.utilities.data_frame import resolve_files, resolve_targets
from flash.core.data.utilities.loading import (
    AUDIO_EXTENSIONS,
    IMG_EXTENSIONS,
    load_data_frame,
    load_spectrogram,
    NP_EXTENSIONS,
)
from flash.core.data.utilities.paths import filter_valid_files, make_dataset, PATH_TYPE
from flash.core.data.utilities.samples import to_samples
from flash.core.utilities.imports import requires


class AudioClassificationInput(Input, ClassificationInputMixin):
    @requires("audio")
    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        h, w = sample[DataKeys.INPUT].shape[-2:]  # H x W
        if DataKeys.METADATA not in sample:
            sample[DataKeys.METADATA] = {}
        sample[DataKeys.METADATA]["size"] = (h, w)
        if DataKeys.TARGET in sample:
            sample[DataKeys.TARGET] = self.format_target(sample[DataKeys.TARGET])
        return sample


class AudioClassificationFilesInput(AudioClassificationInput):
    sampling_rate: int
    n_fft: int

    def load_data(
        self,
        files: List[PATH_TYPE],
        targets: Optional[List[Any]] = None,
        sampling_rate: int = 16000,
        n_fft: int = 400,
        target_formatter: Optional[TargetFormatter] = None,
    ) -> List[Dict[str, Any]]:
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft

        if targets is None:
            files = filter_valid_files(files, valid_extensions=AUDIO_EXTENSIONS + IMG_EXTENSIONS + NP_EXTENSIONS)
            return to_samples(files)
        files, targets = filter_valid_files(
            files, targets, valid_extensions=AUDIO_EXTENSIONS + IMG_EXTENSIONS + NP_EXTENSIONS
        )
        self.load_target_metadata(targets, target_formatter=target_formatter)
        return to_samples(files, targets)

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        filepath = sample[DataKeys.INPUT]
        sample[DataKeys.INPUT] = load_spectrogram(filepath, sampling_rate=self.sampling_rate, n_fft=self.n_fft)
        sample = super().load_sample(sample)
        sample[DataKeys.METADATA]["filepath"] = filepath
        return sample


class AudioClassificationFolderInput(AudioClassificationFilesInput):
    def load_data(
        self,
        folder: PATH_TYPE,
        sampling_rate: int = 16000,
        n_fft: int = 400,
        target_formatter: Optional[TargetFormatter] = None,
    ) -> List[Dict[str, Any]]:
        files, targets = make_dataset(folder, extensions=IMG_EXTENSIONS + NP_EXTENSIONS)
        return super().load_data(
            files, targets, sampling_rate=sampling_rate, n_fft=n_fft, target_formatter=target_formatter
        )


class AudioClassificationNumpyInput(AudioClassificationInput):
    def load_data(
        self, array: Any, targets: Optional[List[Any]] = None, target_formatter: Optional[TargetFormatter] = None
    ) -> List[Dict[str, Any]]:
        if targets is not None:
            self.load_target_metadata(targets, target_formatter=target_formatter)
        return to_samples(array, targets)

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample[DataKeys.INPUT] = np.float32(np.transpose(sample[DataKeys.INPUT], (1, 2, 0)))
        return super().load_sample(sample)


class AudioClassificationTensorInput(AudioClassificationNumpyInput):
    def load_data(
        self, tensor: Any, targets: Optional[List[Any]] = None, target_formatter: Optional[TargetFormatter] = None
    ) -> List[Dict[str, Any]]:
        if targets is not None:
            self.load_target_metadata(targets, target_formatter=target_formatter)
        return to_samples(tensor, targets)

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample[DataKeys.INPUT] = sample[DataKeys.INPUT].numpy()
        return super().load_sample(sample)


class AudioClassificationDataFrameInput(AudioClassificationFilesInput):
    def load_data(
        self,
        data_frame: pd.DataFrame,
        input_key: str,
        target_keys: Optional[Union[str, List[str]]] = None,
        root: Optional[PATH_TYPE] = None,
        resolver: Optional[Callable[[Optional[PATH_TYPE], Any], PATH_TYPE]] = None,
        sampling_rate: int = 16000,
        n_fft: int = 400,
        target_formatter: Optional[TargetFormatter] = None,
    ) -> List[Dict[str, Any]]:
        files = resolve_files(data_frame, input_key, root, resolver)
        if target_keys is not None:
            targets = resolve_targets(data_frame, target_keys)
        else:
            targets = None
        result = super().load_data(
            files, targets, sampling_rate=sampling_rate, n_fft=n_fft, target_formatter=target_formatter
        )

        # If we had binary multi-class targets then we also know the labels (column names)
        if (
            self.training
            and isinstance(self.target_formatter, MultiBinaryTargetFormatter)
            and isinstance(target_keys, List)
        ):
            self.labels = target_keys

        return result


class AudioClassificationCSVInput(AudioClassificationDataFrameInput):
    def load_data(
        self,
        csv_file: PATH_TYPE,
        input_key: str,
        target_keys: Optional[Union[str, List[str]]] = None,
        root: Optional[PATH_TYPE] = None,
        resolver: Optional[Callable[[Optional[PATH_TYPE], Any], PATH_TYPE]] = None,
        sampling_rate: int = 16000,
        n_fft: int = 400,
        target_formatter: Optional[TargetFormatter] = None,
    ) -> List[Dict[str, Any]]:
        data_frame = load_data_frame(csv_file)
        if root is None:
            root = os.path.dirname(csv_file)
        return super().load_data(
            data_frame,
            input_key,
            target_keys,
            root,
            resolver,
            sampling_rate=sampling_rate,
            n_fft=n_fft,
            target_formatter=target_formatter,
        )
