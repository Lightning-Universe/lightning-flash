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


# container to apply augmentations at both image and mask reusing the same parameters
# TODO: we have to figure out how to decide what transforms are applied to mask
# For instance, color transforms cannot be applied to masks
class SegmentationSequential(nn.Sequential):

    def __init__(self, *args):
        super(SegmentationSequential, self).__init__(*args)

    @torch.no_grad()
    def forward(self, img, mask):
        img_out = img.float()
        mask_out = mask[None].float()
        for aug in self.children():
            img_out = aug(img_out)
            # some transforms don't have params
            if hasattr(aug, "_params"):
                mask_out = aug(mask_out, aug._params)
            else:
                mask_out = aug(mask_out)
        return img_out[0], mask_out[0, 0].long()


def default_train_transforms(image_size: Tuple[int, int]) -> Dict[str, Callable]:
    return {
        "post_tensor_transform": SegmentationSequential(
            K.geometry.Resize(image_size, interpolation='nearest'),
            K.augmentation.RandomHorizontalFlip(p=0.75),
        ),
    }


def default_val_transforms(image_size: Tuple[int, int]) -> Dict[str, Callable]:
    return {
        "post_tensor_transform": SegmentationSequential(
            K.geometry.Resize(image_size, interpolation='nearest'),
            K.augmentation.RandomHorizontalFlip(p=0.),  # #TODO: bug somewhere with shapes
        ),
    }
