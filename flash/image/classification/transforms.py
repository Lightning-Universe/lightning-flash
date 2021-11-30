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

from flash.core.data.io.input import DataKeys
from flash.core.data.transforms import ApplyToKeys, kornia_collate, merge_transforms
from flash.core.utilities.imports import _ALBUMENTATIONS_AVAILABLE, _KORNIA_AVAILABLE, _TORCHVISION_AVAILABLE, requires

if _KORNIA_AVAILABLE:
    import kornia as K

if _TORCHVISION_AVAILABLE:
    import torchvision
    from torchvision import transforms as T

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations


class AlbumentationsAdapter(torch.nn.Module):
    @requires("albumentations")
    def __init__(self, transform):
        super().__init__()
        if not isinstance(transform, list):
            transform = [transform]
        self.transform = albumentations.Compose(transform)

    def forward(self, x):
        return torch.from_numpy(self.transform(image=x.numpy())["image"])


def default_transforms(image_size: Tuple[int, int]) -> Dict[str, Callable]:
    """The default transforms for image classification: resize the image, convert the image and target to a tensor,
    collate the batch, and apply normalization."""
    if _KORNIA_AVAILABLE and os.getenv("FLASH_TESTING", "0") != "1":
        #  Better approach as all transforms are applied on tensor directly
        per_sample_transform = T.Compose(
            [
                ApplyToKeys(
                    DataKeys.INPUT,
                    T.Compose([torchvision.transforms.ToTensor(), K.geometry.Resize(image_size)]),
                ),
                ApplyToKeys(DataKeys.TARGET, torch.as_tensor),
            ]
        )
        per_batch_transform_on_device = ApplyToKeys(
            DataKeys.INPUT,
            K.augmentation.Normalize(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])),
        )
        return dict(
            per_sample_transform=per_sample_transform,
            collate=kornia_collate,
            per_batch_transform_on_device=per_batch_transform_on_device,
        )
    return dict(
        per_sample_transform=nn.Sequential(
            ApplyToKeys(
                DataKeys.INPUT,
                nn.Sequential(T.Resize(image_size), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])),
                ApplyToKeys(DataKeys.TARGET, torch.as_tensor),
            )
        ),
        collate=kornia_collate,
    )


def train_default_transforms(image_size: Tuple[int, int]) -> Dict[str, Callable]:
    """During training, we apply the default transforms with additional ``RandomHorizontalFlip``."""
    if _KORNIA_AVAILABLE and os.getenv("FLASH_TESTING", "0") != "1":
        #  Better approach as all transforms are applied on tensor directly
        transforms = {
            "post_tensor_transform": ApplyToKeys(DataKeys.INPUT, K.augmentation.RandomHorizontalFlip()),
        }
    else:
        transforms = {"pre_tensor_transform": ApplyToKeys(DataKeys.INPUT, T.RandomHorizontalFlip())}
    return merge_transforms(default_transforms(image_size), transforms)
