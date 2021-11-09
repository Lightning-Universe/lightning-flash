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

from flash.core.data.data_source import DefaultDataKeys
from flash.core.utilities.imports import _TORCHVISION_AVAILABLE, _VISSL_AVAILABLE
from tests.image.embedding.utils import ssl_datamodule


@pytest.mark.skipif(not (_TORCHVISION_AVAILABLE and _VISSL_AVAILABLE), reason="vissl not installed.")
def test_multicrop_input_transform():
    batch_size = 8
    total_num_crops = 6
    num_crops = [2, 4]
    size_crops = [160, 96]
    crop_scales = [[0.4, 1], [0.05, 0.4]]

    train_dataloader = ssl_datamodule(
        batch_size=batch_size,
        total_num_crops=total_num_crops,
        num_crops=num_crops,
        size_crops=size_crops,
        crop_scales=crop_scales,
    )._train_dataloader()
    batch = next(iter(train_dataloader))

    assert len(batch[DefaultDataKeys.INPUT]) == total_num_crops
    assert batch[DefaultDataKeys.INPUT][0].shape == (batch_size, 3, size_crops[0], size_crops[0])
    assert batch[DefaultDataKeys.INPUT][-1].shape == (batch_size, 3, size_crops[-1], size_crops[-1])
    assert list(batch[DefaultDataKeys.TARGET].shape) == [batch_size]
