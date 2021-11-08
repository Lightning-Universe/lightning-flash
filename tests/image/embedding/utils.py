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
from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.process import DefaultInputTransform
from flash.core.data.transforms import ApplyToKeys
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
    multi_crop_transform = TRANSFORM_REGISTRY["multicrop_ssl_transform"](
        total_num_crops, num_crops, size_crops, crop_scales
    )

    to_tensor_transform = ApplyToKeys(
        DefaultDataKeys.INPUT,
        multi_crop_transform,
    )
    input_transform = DefaultInputTransform(
        train_transform={
            "to_tensor_transform": to_tensor_transform,
            "collate": collate_fn,
        }
    )

    datamodule = ImageClassificationData.from_datasets(
        train_dataset=FakeData(),
        input_transform=input_transform,
        batch_size=batch_size,
    )

    return datamodule
