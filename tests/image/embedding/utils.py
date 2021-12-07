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
from typing import Callable, List, Optional

from flash.core.data.input_transform import InputTransform
from flash.core.utilities.imports import _TORCHVISION_AVAILABLE, _VISSL_AVAILABLE
from flash.image import ImageClassificationData
from flash.image.embedding.vissl.transforms import multicrop_collate_fn

if _TORCHVISION_AVAILABLE:
    from torchvision.datasets import FakeData

if _VISSL_AVAILABLE:
    from classy_vision.dataset.transforms import TRANSFORM_REGISTRY


def ssl_datamodule(
    batch_size=2,
    total_num_crops=4,
    num_crops=[2, 2],
    size_crops=[160, 96],
    crop_scales=[[0.4, 1], [0.05, 0.4]],
    collate_fn=multicrop_collate_fn,
):
    @dataclass
    class SSLInputTransform(InputTransform):

        total_num_crops: int = 4
        num_crops: Optional[List[int]] = None
        size_crops: Optional[List[int]] = None
        crop_scales: Optional[List[List[float]]] = None

        def input_per_sample_transform(self) -> Callable:
            return TRANSFORM_REGISTRY["multicrop_ssl_transform"](
                self.total_num_crops, self.num_crops, self.size_crops, self.crop_scales
            )

        def collate(self) -> Callable:
            return collate_fn

    transform_kwargs = dict(
        total_num_crops=total_num_crops, num_crops=num_crops, size_crops=size_crops, crop_scales=crop_scales
    )

    datamodule = ImageClassificationData.from_datasets(
        train_dataset=FakeData(),
        train_transform=SSLInputTransform,
        val_transform=SSLInputTransform,
        test_transform=SSLInputTransform,
        predict_transform=SSLInputTransform,
        transform_kwargs=transform_kwargs,
        batch_size=batch_size,
    )

    return datamodule
