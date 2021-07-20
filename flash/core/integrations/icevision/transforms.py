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
from typing import Any, Callable, Dict, Sequence, Tuple

from flash.core.utilities.imports import _ICEVISION_AVAILABLE, _TORCHVISION_AVAILABLE, requires_extras

if _TORCHVISION_AVAILABLE:
    pass

if _ICEVISION_AVAILABLE:
    from icevision.tfms import A


def collate(samples: Sequence[Dict[str, Any]]) -> Dict[str, Sequence[Any]]:
    return {key: [sample[key] for sample in samples] for key in samples[0]}


@requires_extras("image")
def default_transforms(image_size: Tuple[int, int]) -> Dict[str, Callable]:
    """The default transforms for object detection: convert the image and targets to a tensor, collate the batch."""
    return {
        # "pre_tensor_transform": ApplyToKeys(
        #     DefaultDataKeys.INPUT,
        #     tfms.A.Adapter([*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()]),
        # )
        "pre_tensor_transform": A.Adapter([*A.resize_and_pad(image_size), A.Normalize()]),
    }


@requires_extras("image")
def train_default_transforms(image_size: Tuple[int, int]) -> Dict[str, Callable]:
    """The default transforms for object detection: convert the image and targets to a tensor, collate the batch."""
    return {
        # "pre_tensor_transform": ApplyToKeys(
        #     DefaultDataKeys.INPUT,
        #     tfms.A.Adapter([*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()]),
        # )
        "pre_tensor_transform": A.Adapter([*A.aug_tfms(size=image_size), A.Normalize()]),
    }
