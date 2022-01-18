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
from typing import Callable, Tuple

import torch

from flash.core.data.io.input_transform import InputTransform
from flash.core.data.transforms import kornia_collate
from flash.core.utilities.imports import _ALBUMENTATIONS_AVAILABLE, _TORCHVISION_AVAILABLE, requires

if _TORCHVISION_AVAILABLE:
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


@dataclass
class ImageClassificationInputTransform(InputTransform):

    image_size: Tuple[int, int] = (196, 196)

    def input_per_sample_transform(self):
        return T.Compose(
            [T.ToTensor(), T.Resize(self.image_size), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

    def train_input_per_sample_transform(self):
        return T.Compose(
            [
                T.ToTensor(),
                T.Resize(self.image_size),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                T.RandomHorizontalFlip(),
            ]
        )

    def target_per_sample_transform(self) -> Callable:
        return torch.as_tensor

    def collate(self) -> Callable:
        # TODO: Remove kornia collate for default_collate
        return kornia_collate
