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
from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn as nn

from flash.core.data.input_transform import InputTransform
from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _KORNIA_AVAILABLE, _TORCHVISION_AVAILABLE

if _KORNIA_AVAILABLE:
    import kornia as K

if _TORCHVISION_AVAILABLE:
    from torchvision.transforms import Compose

    from flash.core.data.transforms import ApplyToKeys, kornia_collate, KorniaParallelTransforms, merge_transforms


def prepare_target(tensor: torch.Tensor) -> torch.Tensor:
    """Convert the target mask to long and remove the channel dimension."""
    return tensor.long().squeeze(1)


def default_transforms(image_size: Tuple[int, int]) -> Dict[str, Callable]:
    """The default transforms for semantic segmentation: resize the image and mask, collate the batch, and apply
    normalization."""
    return {
        "per_sample_transform": nn.Sequential(
            ApplyToKeys(
                [DataKeys.INPUT, DataKeys.TARGET],
                KorniaParallelTransforms(K.geometry.Resize(image_size, interpolation="nearest")),
            ),
        ),
        "collate": Compose([kornia_collate, ApplyToKeys(DataKeys.TARGET, prepare_target)]),
    }


def train_default_transforms(image_size: Tuple[int, int]) -> Dict[str, Callable]:
    """During training, we apply the default transforms with additional ``RandomHorizontalFlip`` and
    ``ColorJitter``."""
    return merge_transforms(
        default_transforms(image_size),
        {
            "per_sample_transform": nn.Sequential(
                ApplyToKeys(
                    [DataKeys.INPUT, DataKeys.TARGET],
                    KorniaParallelTransforms(K.augmentation.RandomHorizontalFlip(p=0.5)),
                ),
            ),
        },
    )


def predict_default_transforms(image_size: Tuple[int, int]) -> Dict[str, Callable]:
    """During predict, we apply the default transforms only on DataKeys.INPUT."""
    return {
        "per_sample_transform": nn.Sequential(
            ApplyToKeys(
                DataKeys.INPUT,
                K.geometry.Resize(image_size, interpolation="nearest"),
            ),
        ),
        "collate": kornia_collate,
    }


def remove_extra_dimensions(batch: Dict[str, Any]):
    if isinstance(batch[DataKeys.INPUT], list):
        batch[DataKeys.INPUT] = batch[DataKeys.INPUT][0]
    return batch


@dataclass
class SemanticSegmentationInputTransform(InputTransform):

    image_size: Tuple[int, int] = (128, 128)

    def train_per_sample_transform(self) -> Callable:
        return ApplyToKeys(
            [DataKeys.INPUT, DataKeys.TARGET],
            KorniaParallelTransforms(
                K.geometry.Resize(self.image_size, interpolation="nearest"), K.augmentation.RandomHorizontalFlip(p=0.5)
            ),
        )

    def per_sample_transform(self) -> Callable:
        return ApplyToKeys(
            [DataKeys.INPUT, DataKeys.TARGET],
            KorniaParallelTransforms(K.geometry.Resize(self.image_size, interpolation="nearest")),
        )

    def predict_input_per_sample_transform(self) -> Callable:
        return K.geometry.Resize(self.image_size, interpolation="nearest")

    def collate(self) -> Callable:
        return kornia_collate

    def target_per_batch_transform(self) -> Callable:
        return prepare_target

    def predict_per_batch_transform(self) -> Callable:
        return remove_extra_dimensions

    def serve_per_batch_transform(self) -> Callable:
        return remove_extra_dimensions
