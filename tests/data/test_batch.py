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
import torch
from torch.tensor import Tensor
from torch.testing import assert_allclose

from flash.data.batch import default_uncollate


class TestDefaultUncollate:

    def test_smoke(self):
        batch = torch.rand(2, 1)
        assert default_uncollate(batch) is not None

    def test_tensor_zero(self):
        batch = torch.tensor(1)
        output = default_uncollate(batch)
        assert_allclose(batch, output)

    def test_tensor_batch(self):
        batch = torch.rand(2, 1)
        output = default_uncollate(batch)
        assert isinstance(output, list)
        assert all([isinstance(x, torch.Tensor) for x in output])

    def test_sequence(self):
        B = 3  # batch_size

        batch = {}
        batch['a'] = torch.rand(B, 4)
        batch['b'] = torch.rand(B, 2)
        batch['c'] = torch.rand(B)

        output = default_uncollate(batch)
        assert isinstance(output, list)
        assert len(batch) == B

        for sample in output:
            assert list(sample.keys()) == ['a', 'b', 'c']
            assert isinstance(sample['a'], list)
            assert len(sample['a']) == 4
            assert isinstance(sample['b'], list)
            assert len(sample['b']) == 2
            assert isinstance(sample['c'], torch.Tensor)
            assert len(sample['c'].shape) == 0
