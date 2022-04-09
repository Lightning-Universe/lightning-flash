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
from functools import partial
from typing import Callable, Optional, Sequence

import torch.nn as nn

from flash.core.registry import FlashRegistry
from flash.image.embedding.vissl.transforms import multicrop_collate_fn, simclr_collate_fn
from flash.image.embedding.vissl.transforms.multicrop import StandardMultiCropSSLTransform


def simclr_transform(
    total_num_crops: int = 2,
    num_crops: Sequence[int] = [2],
    size_crops: Sequence[int] = [224],
    crop_scales: Sequence[Sequence[float]] = [[0.4, 1]],
    gaussian_blur: bool = True,
    jitter_strength: float = 1.0,
    normalize: Optional[nn.Module] = None,
    collate_fn: Callable = simclr_collate_fn,
) -> partial:
    """For simclr and barlow twins."""
    transform = partial(
        StandardMultiCropSSLTransform,
        total_num_crops=total_num_crops,
        num_crops=num_crops,
        size_crops=size_crops,
        crop_scales=crop_scales,
        gaussian_blur=gaussian_blur,
        jitter_strength=jitter_strength,
        normalize=normalize,
        collate_fn=collate_fn,
    )

    return transform


def swav_transform(
    total_num_crops: int = 8,
    num_crops: Sequence[int] = [2, 6],
    size_crops: Sequence[int] = [224, 96],
    crop_scales: Sequence[Sequence[float]] = [[0.4, 1], [0.05, 0.4]],
    gaussian_blur: bool = True,
    jitter_strength: float = 1.0,
    normalize: Optional[nn.Module] = None,
    collate_fn: Callable = multicrop_collate_fn,
) -> partial:
    """For swav."""
    transform = partial(
        StandardMultiCropSSLTransform,
        total_num_crops=total_num_crops,
        num_crops=num_crops,
        size_crops=size_crops,
        crop_scales=crop_scales,
        gaussian_blur=gaussian_blur,
        jitter_strength=jitter_strength,
        normalize=normalize,
        collate_fn=collate_fn,
    )

    return transform


barlow_twins_transform = partial(simclr_transform, collate_fn=simclr_collate_fn)


transforms = [
    "simclr_transform",
    "swav_transform",
    "barlow_twins_transform",
]


def register_vissl_transforms(register: FlashRegistry):
    for idx, transform in enumerate(
        (
            simclr_transform,
            swav_transform,
            barlow_twins_transform,
        )
    ):
        register(transform, name=transforms[idx])
