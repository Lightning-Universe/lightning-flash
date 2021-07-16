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

from flash.audio.classification.transforms import default_transforms, train_default_transforms
from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_module import DataModule
from flash.core.data.data_source import DefaultDataSources
from flash.core.data.process import Deserializer, Preprocess
from flash.core.utilities.imports import requires_extras
from flash.image.classification.data import MatplotlibVisualization
from flash.image.data import ImageDeserializer, ImagePathsDataSource


class AudioClassificationPreprocess(Preprocess):

    @requires_extras(["audio", "image"])
    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]],
        val_transform: Optional[Dict[str, Callable]],
        test_transform: Optional[Dict[str, Callable]],
        predict_transform: Optional[Dict[str, Callable]],
        spectrogram_size: Tuple[int, int] = (196, 196),
        time_mask_param: int = 80,
        freq_mask_param: int = 80,
        deserializer: Optional['Deserializer'] = None,
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
                DefaultDataSources.FILES: ImagePathsDataSource(),
                DefaultDataSources.FOLDERS: ImagePathsDataSource()
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


class AudioClassificationData(DataModule):
    """Data module for audio classification."""

    preprocess_cls = AudioClassificationPreprocess

    def set_block_viz_window(self, value: bool) -> None:
        """Setter method to switch on/off matplotlib to pop up windows."""
        self.data_fetcher.block_viz_window = value

    @staticmethod
    def configure_data_fetcher(*args, **kwargs) -> BaseDataFetcher:
        return MatplotlibVisualization(*args, **kwargs)
