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
from torch.utils.data.dataloader import default_collate

from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.transforms import ApplyToKeys, kornia_collate, merge_transforms
from flash.core.data_v2.preprocess_transform import PreprocessTransform, PreprocessTransformPlacement
from flash.core.registry import FlashRegistry
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
        return {
            "per_sample_transform": nn.Sequential(
                ApplyToKeys(
                    DefaultDataKeys.INPUT,
                    T.Compose(
                        [
                            torchvision.transforms.ToTensor(),
                            K.geometry.Resize(image_size),
                            K.augmentation.Normalize(
                                torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
                            ),
                        ]
                    ),
                ),
                ApplyToKeys(DefaultDataKeys.TARGET, torch.as_tensor),
            ),
            "collate": kornia_collate,
            "per_batch_transform_on_device": ApplyToKeys(
                DefaultDataKeys.INPUT,
                K.augmentation.Normalize(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])),
            ),
        }
    return {
        "per_sample_transform": nn.Sequential(
            ApplyToKeys(
                DefaultDataKeys.INPUT,
                T.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        T.Resize(image_size),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]
                ),
            ),
            ApplyToKeys(DefaultDataKeys.TARGET, torch.as_tensor),
        ),
        "collate": kornia_collate,
    }


def train_default_transforms(image_size: Tuple[int, int]) -> Dict[str, Callable]:
    """During training, we apply the default transforms with additional ``RandomHorizontalFlip``."""
    if _KORNIA_AVAILABLE and os.getenv("FLASH_TESTING", "0") != "1":
        #  Better approach as all transforms are applied on tensor directly
        transforms = {
            "per_sample_transform": ApplyToKeys(DefaultDataKeys.INPUT, K.augmentation.RandomHorizontalFlip()),
        }
    else:
        transforms = {"per_sample_transform": ApplyToKeys(DefaultDataKeys.INPUT, T.RandomHorizontalFlip())}
    return merge_transforms(default_transforms(image_size), transforms)


class DefaultImageClassificationPreprocessTransform(PreprocessTransform):
    def configure_transforms(
        self, image_size: Tuple[int, int] = (196, 196)
    ) -> Dict[PreprocessTransformPlacement, Callable]:
        if _KORNIA_AVAILABLE and os.getenv("FLASH_TESTING", "0") != "1":
            #  Better approach as all transforms are applied on tensor directly
            return {
                PreprocessTransformPlacement.PER_SAMPLE_TRANSFORM: nn.Sequential(
                    ApplyToKeys(
                        DefaultDataKeys.INPUT,
                        T.Compose(
                            [
                                torchvision.transforms.ToTensor(),
                                K.geometry.Resize(image_size),
                                K.augmentation.Normalize(
                                    torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
                                ),
                            ]
                            + (self.training) * [K.augmentation.RandomHorizontalFlip()]
                        ),
                    ),
                    ApplyToKeys(DefaultDataKeys.TARGET, torch.as_tensor),
                ),
                PreprocessTransformPlacement.COLLATE: kornia_collate,
                PreprocessTransformPlacement.PER_BATCH_TRANSFORM_ON_DEVICE: ApplyToKeys(
                    DefaultDataKeys.INPUT,
                    K.augmentation.Normalize(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])),
                ),
            }
        return {
            PreprocessTransformPlacement.PER_SAMPLE_TRANSFORM: nn.Sequential(
                ApplyToKeys(
                    DefaultDataKeys.INPUT,
                    T.Compose(
                        [
                            torchvision.transforms.ToTensor(),
                            T.Resize(image_size),
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        ]
                        + (self.training) * [T.RandomHorizontalFlip()]
                    ),
                ),
                ApplyToKeys(DefaultDataKeys.TARGET, torch.as_tensor),
            ),
            PreprocessTransformPlacement.COLLATE: default_collate,
        }


IMAGE_CLASSIFICATION_REGISTRY = FlashRegistry("transforms")
IMAGE_CLASSIFICATION_REGISTRY(name="default", fn=DefaultImageClassificationPreprocessTransform)
