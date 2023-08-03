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
import torch
from flash.core.data.io.output_transform import OutputTransform
from flash.core.utilities.imports import _TOPIC_CORE_AVAILABLE


@pytest.mark.skipif(not _TOPIC_CORE_AVAILABLE, reason="Not testing core.")
def test_output_transform():
    class CustomOutputTransform(OutputTransform):
        @staticmethod
        def per_batch_transform(batch):
            return batch * 2

        @staticmethod
        def per_sample_transform(sample):
            return sample + 1

    output_transform = CustomOutputTransform()
    transformed = output_transform(torch.ones(10))
    assert all(torch.isclose(t, torch.tensor(3.0)) for t in transformed)
