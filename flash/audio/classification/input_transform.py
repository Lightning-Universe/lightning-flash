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
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils.data._utils.collate import default_collate

from flash.core.data.input_transform import InputTransform
from flash.core.data.io.input import DataKeys
from flash.core.data.transforms import ApplyToKeys, merge_transforms
from flash.core.utilities.imports import _TORCHAUDIO_AVAILABLE, _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as T

if _TORCHAUDIO_AVAILABLE:
    from torchaudio import transforms as TAudio


def default_transforms(spectrogram_size: Tuple[int, int]) -> Dict[str, Callable]:
    """The default transforms for audio classification for spectrograms: resize the spectrogram, convert the
    spectrogram and target to a tensor, and collate the batch."""
    return {
        "per_sample_transform": nn.Sequential(
            ApplyToKeys(DataKeys.INPUT, T.Compose([T.ToTensor(), T.Resize(spectrogram_size)])),
            ApplyToKeys(DataKeys.TARGET, torch.as_tensor),
        ),
        "collate": default_collate,
    }


def train_default_transforms(
    spectrogram_size: Tuple[int, int], time_mask_param: Optional[int], freq_mask_param: Optional[int]
) -> Dict[str, Callable]:
    """During training we apply the default transforms with optional ``TimeMasking`` and ``Frequency Masking``."""
    augs = []

    if time_mask_param is not None:
        augs.append(ApplyToKeys(DataKeys.INPUT, TAudio.TimeMasking(time_mask_param=time_mask_param)))

    if freq_mask_param is not None:
        augs.append(ApplyToKeys(DataKeys.INPUT, TAudio.FrequencyMasking(freq_mask_param=freq_mask_param)))

    if len(augs) > 0:
        return merge_transforms(default_transforms(spectrogram_size), {"per_sample_transform": nn.Sequential(*augs)})
    return default_transforms(spectrogram_size)


@dataclass
class AudioClassificationInputTransform(InputTransform):

    spectrogram_size: Tuple[int, int] = (128, 128)
    time_mask_param: Optional[int] = None
    freq_mask_param: Optional[int] = None

    def train_input_per_sample_transform(self) -> Callable:
        transforms = []
        if self.time_mask_param is not None:
            transforms.append(TAudio.TimeMasking(time_mask_param=self.time_mask_param))

        if self.freq_mask_param is not None:
            transforms.append(TAudio.FrequencyMasking(freq_mask_param=self.freq_mask_param))

        transforms += [T.ToTensor(), T.Resize(self.spectrogram_size)]
        return T.Compose(transforms)

    def input_per_sample_transform(self) -> Callable:
        return T.Compose([T.ToTensor(), T.Resize(self.spectrogram_size)])

    def target_per_sample_transform(self) -> Callable:
        return torch.as_tensor
