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
from typing import Any, Callable, Dict, Tuple, Union

from flash.core.data.io.input import DataKeys
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.transforms import AlbumentationsAdapter
from flash.core.utilities.imports import _ALBUMENTATIONS_AVAILABLE

if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as alb
else:
    alb = None


def prepare_target(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Convert the target mask to long and remove the channel dimension."""
    if DataKeys.TARGET in batch:
        batch[DataKeys.TARGET] = batch[DataKeys.TARGET].long().squeeze(1)
    return batch


def remove_extra_dimensions(batch: Dict[str, Any]):
    if isinstance(batch[DataKeys.INPUT], list):
        assert len(batch[DataKeys.INPUT]) == 1
        batch[DataKeys.INPUT] = batch[DataKeys.INPUT][0]
    return batch


@dataclass
class SemanticSegmentationInputTransform(InputTransform):
    # https://albumentations.ai/docs/examples/pytorch_semantic_segmentation

    image_size: Tuple[int, int] = (128, 128)
    image_color_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_color_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    def train_per_sample_transform(self) -> Callable:
        return AlbumentationsAdapter(
            [
                alb.Resize(*self.image_size),
                alb.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                alb.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
                alb.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                alb.Normalize(mean=self.image_color_mean, std=self.image_color_std),
            ]
        )

    def per_sample_transform(self) -> Callable:
        return AlbumentationsAdapter(
            [
                alb.Resize(*self.image_size),
                alb.Normalize(mean=self.image_color_mean, std=self.image_color_std),
            ]
        )

    def predict_input_per_sample_transform(self) -> Callable:
        return AlbumentationsAdapter(
            [
                alb.Resize(*self.image_size),
                alb.Normalize(mean=self.image_color_mean, std=self.image_color_std),
            ]
        )

    def per_batch_transform(self) -> Callable:
        return T.Compose([prepare_target, remove_extra_dimensions])
