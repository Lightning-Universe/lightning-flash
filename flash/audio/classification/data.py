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
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from flash.audio.classification.transforms import default_transforms, train_default_transforms
from flash.core.data.data_source import (
    DefaultDataSources,
    has_file_allowed_extension,
    LoaderDataFrameDataSource,
    PathsDataSource,
)
from flash.core.data.process import Deserializer, Preprocess
from flash.core.utilities.imports import _TORCHVISION_AVAILABLE
from flash.image.classification.data import ImageClassificationData
from flash.image.data import ImageDeserializer, IMG_EXTENSIONS, NP_EXTENSIONS

if _TORCHVISION_AVAILABLE:
    from torchvision.datasets.folder import default_loader


def spectrogram_loader(filepath: str):
    if has_file_allowed_extension(filepath, IMG_EXTENSIONS):
        img = default_loader(filepath)
        data = np.array(img)
    else:
        data = np.load(filepath)
    return data


class AudioClassificationPathsDataSource(PathsDataSource):
    def __init__(self):
        super().__init__(loader=spectrogram_loader, extensions=IMG_EXTENSIONS + NP_EXTENSIONS)


class AudioClassificationDataFrameDataSource(LoaderDataFrameDataSource):
    def __init__(self):
        super().__init__(spectrogram_loader)


class AudioClassificationPreprocess(Preprocess):
    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        spectrogram_size: Tuple[int, int] = (128, 128),
        time_mask_param: int = 80,
        freq_mask_param: int = 80,
        deserializer: Optional["Deserializer"] = None,
    ):
        self.spectrogram_size = spectrogram_size
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources={
                DefaultDataSources.FILES: AudioClassificationPathsDataSource(),
                DefaultDataSources.FOLDERS: AudioClassificationPathsDataSource(),
                "data_frame": AudioClassificationDataFrameDataSource(),
                DefaultDataSources.CSV: AudioClassificationDataFrameDataSource(),
            },
            deserializer=deserializer or ImageDeserializer(),
            default_data_source=DefaultDataSources.FILES,
        )

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            **self.transforms,
            "spectrogram_size": self.spectrogram_size,
            "time_mask_param": self.time_mask_param,
            "freq_mask_param": self.freq_mask_param,
        }

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = False):
        return cls(**state_dict)

    def default_transforms(self) -> Optional[Dict[str, Callable]]:
        return default_transforms(self.spectrogram_size)

    def train_default_transforms(self) -> Optional[Dict[str, Callable]]:
        return train_default_transforms(self.spectrogram_size, self.time_mask_param, self.freq_mask_param)


class AudioClassificationData(ImageClassificationData):
    """Data module for audio classification."""

    preprocess_cls = AudioClassificationPreprocess
