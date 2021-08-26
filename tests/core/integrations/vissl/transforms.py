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
import pytest

from flash.image import ImageClassificationData
from flash.core.utilities.imports import _IMAGE_AVAILABLE
from flash.core.data.transforms import ApplyToKeys
from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.process import DefaultPreprocess

if _IMAGE_AVAILABLE:
    from torchvision.datasets import CIFAR10
    from classy_vision.dataset.transforms import TRANSFORM_REGISTRY
    from flash.core.integrations.vissl.transforms import vissl_collate_fn


@pytest.mark.skipif(not _IMAGE_AVAILABLE, reason="vissl not installed.")
def test_multicrop_input_transform():
    batch_size = 8
    total_crops = 6
    num_crops = [2, 4]
    size_crops = [160, 96]
    crop_scales = [[0.4, 1], [0.05, 0.4]]

    multi_crop_transform = TRANSFORM_REGISTRY['multicrop_ssl_transform'](
        total_crops, num_crops, size_crops, crop_scales
    )

    to_tensor_transform = ApplyToKeys(
        DefaultDataKeys.INPUT,
        multi_crop_transform,
    )
    preprocess = DefaultPreprocess(
        train_transform={
            'to_tensor_transform': to_tensor_transform,
            'collate': vissl_collate_fn,
        }
    )

    datamodule = ImageClassificationData.from_datasets(
        train_dataset=CIFAR10('.', download=True),
        preprocess=preprocess,
        batch_size=batch_size,
    )

    train_dataloader = datamodule._train_dataloader()
    batch = next(iter(train_dataloader))

    assert len(batch[DefaultDataKeys.INPUT]) == total_crops
    assert batch[DefaultDataKeys.INPUT][0].shape == (batch_size, 3, size_crops[0], size_crops[0])
    assert batch[DefaultDataKeys.INPUT][-1].shape == (batch_size, 3, size_crops[-1], size_crops[-1])
    assert list(batch[DefaultDataKeys.TARGET].shape) == [batch_size]
