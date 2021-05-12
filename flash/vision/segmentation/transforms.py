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
from typing import Callable, Dict, Tuple

import kornia as K
import torch
import torch.nn as nn

from flash.data.data_source import DefaultDataKeys
from flash.data.transforms import ApplyToKeys, kornia_collate, KorniaParallelTransforms, merge_transforms


def prepare_target(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.long().squeeze()


def default_transforms(image_size: Tuple[int, int]) -> Dict[str, Callable]:
    """The default transforms for semantic segmentation: resize the image and mask, collate the batch, and apply
    normalization."""
    return {
        "post_tensor_transform": nn.Sequential(
            ApplyToKeys(
                [DefaultDataKeys.INPUT, DefaultDataKeys.TARGET],
                KorniaParallelTransforms(K.geometry.Resize(image_size, interpolation='nearest')),
            ),
        ),
        "collate": nn.Sequential(ApplyToKeys(DefaultDataKeys.TARGET, prepare_target), kornia_collate),
        "per_batch_transform_on_device": ApplyToKeys(DefaultDataKeys.INPUT, K.enhance.Normalize(0., 255.)),
    }


def train_default_transforms(image_size: Tuple[int, int]) -> Dict[str, Callable]:
    """During training, we apply the default transforms with additional ``RandomHorizontalFlip`` and ``ColorJitter``."""
    return merge_transforms(
        default_transforms(image_size), {
            "post_tensor_transform": nn.Sequential(
                ApplyToKeys(
                    [DefaultDataKeys.INPUT, DefaultDataKeys.TARGET],
                    KorniaParallelTransforms(K.augmentation.RandomHorizontalFlip(p=0.75)),
                ),
            ),
            "per_batch_transform_on_device": ApplyToKeys(
                DefaultDataKeys.INPUT,
                K.augmentation.ColorJitter(0.4, p=0.5),
            ),
        }
    )
