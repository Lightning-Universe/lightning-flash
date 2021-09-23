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
from typing import List, Optional, Sequence

import numpy as np
import torch.nn as nn

from flash.core.utilities.imports import _TORCHVISION_AVAILABLE, Image

if _TORCHVISION_AVAILABLE:
    import torchvision.transforms as pth_transforms


class StandardMultiCropSSLTransform(nn.Module):
    """Convert a PIL image to Multi-resolution Crops. The input is a PIL image and output is the list of image
    crops.

    This transform was proposed in SwAV - https://arxiv.org/abs/2006.09882
    This transform can act as a base transform class for SimCLR, SwAV, MoCo, Barlow Twins and DINO from VISSL.

    This transform has been modified from the ImgPilToMultiCrop code present at
    https://github.com/facebookresearch/vissl/blob/master/vissl/data/ssl_transforms/img_pil_to_multicrop.py
    """

    def __init__(
        self,
        total_num_crops: int,
        num_crops: Sequence[int],
        size_crops: Sequence[int],
        crop_scales: Sequence[Sequence[float]],
        gaussian_blur: bool = True,
        jitter_strength: float = 1.0,
        normalize: Optional[nn.Module] = None,
    ):
        """Returns total_num_crops square crops of an image. Each crop is a random crop extracted according to the
        parameters specified in size_crops and crop_scales. For ease of use, one can specify `num_crops` which
        removes the need to repeat parameters.

        Args:
            total_num_crops (int): Total number of crops to extract
            num_crops (List or Tuple of ints): Specifies the number of `type' of crops.
            size_crops (List or Tuple of ints): Specifies the height (height = width)
                                                of each patch
            crop_scales (List or Tuple containing [float, float]): Scale of the crop
            gaussian_blur (bool): Specifies if the transforms composition has Gaussian Blur
            jitter_strength (float): Specify the coefficient for color jitter transform
            normalize (Optional): Normalize transform from torchvision with params set
                                  according to the dataset

        Example usage:
        - (total_num_crops=2, num_crops=[1, 1],
           size_crops=[224, 96], crop_scales=[(0.14, 1.), (0.05, 0.14)])
           Extracts 2 crops total of size 224x224 and 96x96
        - (total_num_crops=3, num_crops=[1, 2],
           size_crops=[224, 96], crop_scales=[(0.14, 1.), (0.05, 0.14)])
           Extracts 3 crops total: 1 of size 224x224 and 2 of size 96x96
        """
        super().__init__()

        assert np.sum(num_crops) == total_num_crops
        assert len(size_crops) == len(num_crops)
        assert len(size_crops) == len(crop_scales)

        self.gaussian_blur = gaussian_blur
        self.jitter_strength = jitter_strength
        self.normalize = normalize

        color_jitter = pth_transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )
        color_transform = [pth_transforms.RandomApply([color_jitter], p=0.8), pth_transforms.RandomGrayscale(p=0.2)]

        if self.gaussian_blur:
            kernel_size = int(0.1 * size_crops[0])
            if kernel_size % 2 == 0:
                kernel_size += 1

            color_transform.append(
                pth_transforms.RandomApply([pth_transforms.GaussianBlur(kernel_size=kernel_size)], p=0.5)
            )

        self.color_transform = pth_transforms.Compose(color_transform)

        if normalize is None:
            self.final_transform = pth_transforms.ToTensor()
        else:
            self.final_transform = pth_transforms.Compose([pth_transforms.ToTensor(), normalize])

        transforms = []
        for num, size, scale in zip(num_crops, size_crops, crop_scales):
            transforms.extend(
                [
                    pth_transforms.Compose(
                        [
                            pth_transforms.RandomResizedCrop(size, scale=scale),
                            pth_transforms.RandomHorizontalFlip(p=0.5),
                            self.color_transform,
                            self.final_transform,
                        ]
                    )
                ]
                * num
            )

        self.transforms = transforms

    def __call__(self, image: Image.Image) -> List[Image.Image]:
        images = [transform(image) for transform in self.transforms]
        return images
