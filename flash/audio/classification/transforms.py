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
from typing import Callable, Dict, Tuple

import torch
from torch import nn

from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.transforms import ApplyToKeys, kornia_collate, merge_transforms
from flash.core.utilities.imports import _AUDIO_AVAILABLE, _KORNIA_AVAILABLE, _TORCHVISION_AVAILABLE

if _KORNIA_AVAILABLE:
    import kornia as K

if _TORCHVISION_AVAILABLE:
    import torchvision
    from torchvision import transforms as T

if _AUDIO_AVAILABLE:
    from torchaudio import transforms as TAudio


def default_transforms(spectrogram_size: Tuple[int, int]) -> Dict[str, Callable]:
    """The default transforms for audio classification for spectrograms: resize the spectrogram,
    convert the spectrogram and target to a tensor, and collate the batch."""
    if _KORNIA_AVAILABLE and os.getenv("FLASH_TESTING", "0") != "1":
        #  Better approach as all transforms are applied on tensor directly
        return {
            "to_tensor_transform": nn.Sequential(
                ApplyToKeys(DefaultDataKeys.INPUT, torchvision.transforms.ToTensor()),
                ApplyToKeys(DefaultDataKeys.TARGET, torch.as_tensor),
            ),
            "post_tensor_transform": ApplyToKeys(
                DefaultDataKeys.INPUT,
                K.geometry.Resize(spectrogram_size),
            ),
            "collate": kornia_collate,
        }
    return {
        "pre_tensor_transform": ApplyToKeys(DefaultDataKeys.INPUT, T.Resize(spectrogram_size)),
        "to_tensor_transform": nn.Sequential(
            ApplyToKeys(DefaultDataKeys.INPUT, torchvision.transforms.ToTensor()),
            ApplyToKeys(DefaultDataKeys.TARGET, torch.as_tensor),
        ),
        "collate": kornia_collate,
    }


def train_default_transforms(spectrogram_size: Tuple[int, int], time_mask_param: int,
                             freq_mask_param: int) -> Dict[str, Callable]:
    """During training we apply the default transforms with aditional ``TimeMasking`` and ``Frequency Masking``"""
    if os.getenv("FLASH_TESTING", "0") != 1:
        transforms = {
            "post_tensor_transform": nn.Sequential(
                ApplyToKeys(DefaultDataKeys.INPUT, TAudio.TimeMasking(time_mask_param=time_mask_param)),
                ApplyToKeys(DefaultDataKeys.INPUT, TAudio.FrequencyMasking(freq_mask_param=freq_mask_param))
            )
        }

    return merge_transforms(default_transforms(spectrogram_size), transforms)
