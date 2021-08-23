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
from flash.core.utilities.imports import _KORNIA_AVAILABLE, _TORCHVISION_AVAILABLE

if _KORNIA_AVAILABLE:
    import kornia as K

if _TORCHVISION_AVAILABLE:
    import torchvision
    from torchvision import transforms as T


def default_transforms(image_size: Tuple[int, int]) -> Dict[str, Callable]:
    """The default transforms for image classification: resize the image, convert the image and target to a tensor,
    collate the batch, and apply normalization."""
    if _KORNIA_AVAILABLE and os.getenv("FLASH_TESTING", "0") != "1":
        #  Better approach as all transforms are applied on tensor directly
        return {
            "to_tensor_transform": nn.Sequential(
                ApplyToKeys(DefaultDataKeys.INPUT, torchvision.transforms.ToTensor()),
                ApplyToKeys(DefaultDataKeys.TARGET, torch.as_tensor),
            ),
            "post_tensor_transform": ApplyToKeys(
                DefaultDataKeys.INPUT,
                K.geometry.Resize(image_size),
            ),
            "collate": kornia_collate,
            "per_batch_transform_on_device": ApplyToKeys(
                DefaultDataKeys.INPUT,
                K.augmentation.Normalize(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])),
            ),
        }
    return {
        "pre_tensor_transform": ApplyToKeys(DefaultDataKeys.INPUT, T.Resize(image_size)),
        "to_tensor_transform": nn.Sequential(
            ApplyToKeys(DefaultDataKeys.INPUT, torchvision.transforms.ToTensor()),
            ApplyToKeys(DefaultDataKeys.TARGET, torch.as_tensor),
        ),
        "post_tensor_transform": ApplyToKeys(
            DefaultDataKeys.INPUT,
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ),
        "collate": kornia_collate,
    }


def train_default_transforms(image_size: Tuple[int, int]) -> Dict[str, Callable]:
    """During training, we apply the default transforms with additional ``RandomHorizontalFlip``."""
    if _KORNIA_AVAILABLE and os.getenv("FLASH_TESTING", "0") != "1":
        #  Better approach as all transforms are applied on tensor directly
        transforms = {
            "post_tensor_transform": ApplyToKeys(DefaultDataKeys.INPUT, K.augmentation.RandomHorizontalFlip()),
        }
    else:
        transforms = {"pre_tensor_transform": ApplyToKeys(DefaultDataKeys.INPUT, T.RandomHorizontalFlip())}
    return merge_transforms(default_transforms(image_size), transforms)
