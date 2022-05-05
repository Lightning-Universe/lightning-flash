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
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence

import torch.nn as nn
from torch.utils.data._utils.collate import default_collate

from flash.core.data.io.input import DataKeys
from flash.core.data.io.input_transform import InputTransform
from flash.core.utilities.imports import _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    import torchvision.transforms as pth_transforms


@dataclass
class StandardMultiCropSSLTransform(InputTransform):
    """Convert a PIL image to Multi-resolution Crops. The input is a PIL image and output is the list of image
    crops.

    This transform was proposed in SwAV - https://arxiv.org/abs/2006.09882
    This transform can act as a base transform class for SimCLR, SwAV, and Barlow Twins from VISSL.

    This transform has been modified from the ImgPilToMultiCrop code present at
    https://github.com/facebookresearch/vissl/blob/master/vissl/data/ssl_transforms/img_pil_to_multicrop.py

    Args:
        total_num_crops (int): Total number of crops to extract
        num_crops (List or Tuple of ints): Specifies the number of `type' of crops.
        size_crops (List or Tuple of ints): Specifies the height (height = width) of each patch
        crop_scales (List or Tuple containing [float, float]): Scale of the crop
        gaussian_blur (bool): Specifies if the transforms' composition has Gaussian Blur
        jitter_strength (float): Specify the coefficient for color jitter transform
        normalize (Optional): Normalize transform from torchvision with params set according to the dataset
    """

    total_num_crops: int = 2
    num_crops: Sequence[int] = field(default_factory=lambda: [2])
    size_crops: Sequence[int] = field(default_factory=lambda: [224])
    crop_scales: Sequence[Sequence[float]] = field(default_factory=lambda: [[0.4, 1]])
    gaussian_blur: bool = True
    jitter_strength: float = 1.0
    normalize: Optional[nn.Module] = None
    collate_fn: Callable = default_collate

    @staticmethod
    def _apply(transform, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample[DataKeys.INPUT] = transform(sample[DataKeys.INPUT])
        return sample

    @staticmethod
    def _parallel_apply(transforms, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample[DataKeys.INPUT] = [transform(sample[DataKeys.INPUT]) for transform in transforms]
        return sample

    def _get_final_transform(self) -> Callable:
        if self.normalize is None:
            final_transform = pth_transforms.ToTensor()
        else:
            final_transform = pth_transforms.Compose([pth_transforms.ToTensor(), self.normalize])
        return final_transform

    def per_sample_transform(self) -> Callable:
        color_jitter = pth_transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )
        color_transform = [pth_transforms.RandomApply([color_jitter], p=0.8), pth_transforms.RandomGrayscale(p=0.2)]

        if self.gaussian_blur:
            kernel_size = int(0.1 * self.size_crops[0])
            if kernel_size % 2 == 0:
                kernel_size += 1

            color_transform.append(
                pth_transforms.RandomApply([pth_transforms.GaussianBlur(kernel_size=kernel_size)], p=0.5)
            )

        color_transform = pth_transforms.Compose(color_transform)

        final_transform = self._get_final_transform()

        transforms = []
        for num, size, scale in zip(self.num_crops, self.size_crops, self.crop_scales):
            transforms.extend(
                [
                    pth_transforms.Compose(
                        [
                            pth_transforms.RandomResizedCrop(size, scale=scale),
                            pth_transforms.RandomHorizontalFlip(p=0.5),
                            color_transform,
                            final_transform,
                        ]
                    )
                ]
                * num
            )

        return partial(self._parallel_apply, transforms)

    def collate(self) -> Callable:
        return self.collate_fn

    def predict_per_sample_transform(self) -> Callable:
        return partial(
            self._apply,
            pth_transforms.Compose([pth_transforms.CenterCrop(self.size_crops[0]), self._get_final_transform()]),
        )

    def predict_collate(self) -> Callable:
        return default_collate
