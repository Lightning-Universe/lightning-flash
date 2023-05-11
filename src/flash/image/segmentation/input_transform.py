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

from flash.core.data.io.input import DataKeys
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.transforms import AlbumentationsAdapter, ApplyToKeys
from flash.core.utilities.imports import _ALBUMENTATIONS_AVAILABLE, _TORCHVISION_AVAILABLE, requires

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as alb
else:
    alb = None

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as T


def prepare_target(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Convert the target mask to long and remove the channel dimension."""
    if DataKeys.TARGET in batch:
        batch[DataKeys.TARGET] = batch[DataKeys.TARGET].squeeze().long()
    return batch


def permute_target(sample: Dict[str, Any]) -> Dict[str, Any]:
    if DataKeys.TARGET in sample:
        target = sample[DataKeys.TARGET]
        if target.ndim == 2:
            target = target[None, :, :]
        sample[DataKeys.TARGET] = target.transpose((1, 2, 0))
    return sample


def remove_extra_dimensions(batch: Dict[str, Any]):
    if isinstance(batch[DataKeys.INPUT], list):
        assert len(batch[DataKeys.INPUT]) == 1
        batch[DataKeys.INPUT] = batch[DataKeys.INPUT][0]
    return batch


@dataclass
class SemanticSegmentationInputTransform(InputTransform):
    # https://albumentations.ai/docs/examples/pytorch_semantic_segmentation

    image_size: Tuple[int, int] = (128, 128)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    @requires("image")
    def train_per_sample_transform(self) -> Callable:
        return T.Compose(
            [
                permute_target,
                AlbumentationsAdapter(
                    [
                        alb.Resize(*self.image_size),
                        alb.HorizontalFlip(p=0.5),
                        alb.Normalize(mean=self.mean, std=self.std),
                    ]
                ),
                ApplyToKeys(
                    DataKeys.INPUT,
                    T.ToTensor(),
                ),
            ]
        )

    @requires("image")
    def per_sample_transform(self) -> Callable:
        return T.Compose(
            [
                permute_target,
                AlbumentationsAdapter(
                    [
                        alb.Resize(*self.image_size),
                        alb.Normalize(mean=self.mean, std=self.std),
                    ]
                ),
                ApplyToKeys(
                    DataKeys.INPUT,
                    T.ToTensor(),
                ),
            ]
        )

    def per_batch_transform(self) -> Callable:
        return T.Compose([prepare_target, remove_extra_dimensions])
