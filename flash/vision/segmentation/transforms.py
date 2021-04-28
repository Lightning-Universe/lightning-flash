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
from typing import Callable, Dict, List, Tuple

import kornia as K
import torch
import torch.nn as nn

from flash.vision.segmentation.serialization import SegmentationKeys


class ApplyTransformToKeys(nn.Sequential):

    def __init__(self, keys: List[str], *args):
        super().__init__(*args)
        self.keys = keys

    @torch.no_grad()
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for seq in self.children():
            for aug in seq:
                for key in self.keys:
                    # kornia caches the random parameters in `_params` after every
                    # forward call in the augmentation object. We check whether the
                    # random parameters have been generated so that we can apply the
                    # same parameters to images and masks.
                    if hasattr(aug, "_params") and bool(aug._params):
                        params = aug._params
                        x[key] = aug(x[key], params)
                    else:  # case for non random transforms
                        x[key] = aug(x[key])
        return x


def default_train_transforms(image_size: Tuple[int, int]) -> Dict[str, Callable]:
    return {
        "post_tensor_transform": nn.Sequential(
            ApplyTransformToKeys([SegmentationKeys.IMAGES, SegmentationKeys.MASKS],
                                 nn.Sequential(
                                     K.geometry.Resize(image_size, interpolation='nearest'),
                                     K.augmentation.RandomHorizontalFlip(p=0.75),
                                 )),
            ApplyTransformToKeys(
                [SegmentationKeys.IMAGES],
                nn.Sequential(
                    K.enhance.Normalize(0., 255.),
                    K.augmentation.ColorJitter(0.4, p=0.5),
                    # NOTE: uncomment to visualise better
                    # K.enhance.Denormalize(0., 255.),
                )
            ),
        ),
    }


def default_val_transforms(image_size: Tuple[int, int]) -> Dict[str, Callable]:
    return {
        "post_tensor_transform": nn.Sequential(
            ApplyTransformToKeys([SegmentationKeys.IMAGES, SegmentationKeys.MASKS],
                                 nn.Sequential(K.geometry.Resize(image_size, interpolation='nearest'), )),
            ApplyTransformToKeys([SegmentationKeys.IMAGES], nn.Sequential(K.enhance.Normalize(0., 255.), )),
        ),
    }
