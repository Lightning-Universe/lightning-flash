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
import torchvision
from torch import nn
from torchvision import transforms as T

from flash.data.data_source import DefaultDataKeys
from flash.data.transforms import ApplyToKeys
from flash.utils.imports import _KORNIA_AVAILABLE

if _KORNIA_AVAILABLE:
    import kornia as K


def default_train_transforms(image_size: Tuple[int, int]) -> Dict[str, Callable]:
    if _KORNIA_AVAILABLE and not os.getenv("FLASH_TESTING", "0") == "1":
        #  Better approach as all transforms are applied on tensor directly
        return {
            "to_tensor_transform": nn.Sequential(
                ApplyToKeys(DefaultDataKeys.INPUT, torchvision.transforms.ToTensor()),
                ApplyToKeys(DefaultDataKeys.TARGET, torch.as_tensor),
            ),
            "post_tensor_transform": ApplyToKeys(
                DefaultDataKeys.INPUT,
                K.geometry.Resize(image_size),
                K.augmentation.RandomHorizontalFlip(),
            ),
            "per_batch_transform_on_device": ApplyToKeys(
                DefaultDataKeys.INPUT,
                K.augmentation.Normalize(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])),
            )
        }
    else:
        return {
            "pre_tensor_transform": ApplyToKeys(DefaultDataKeys.INPUT, T.Resize(image_size), T.RandomHorizontalFlip()),
            "to_tensor_transform": nn.Sequential(
                ApplyToKeys(DefaultDataKeys.INPUT, torchvision.transforms.ToTensor()),
                ApplyToKeys(DefaultDataKeys.TARGET, torch.as_tensor),
            ),
            "post_tensor_transform": ApplyToKeys(
                DefaultDataKeys.INPUT,
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ),
        }


def default_val_transforms(image_size: Tuple[int, int]) -> Dict[str, Callable]:
    if _KORNIA_AVAILABLE and not os.getenv("FLASH_TESTING", "0") == "1":
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
            "per_batch_transform_on_device": ApplyToKeys(
                DefaultDataKeys.INPUT,
                K.augmentation.Normalize(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])),
            )
        }
    else:
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
        }
