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
from collections import namedtuple
from unittest.mock import Mock

import torch
from pytorch_lightning.trainer.states import RunningStage
from torch.testing import assert_allclose
from torch.utils.data._utils.collate import default_collate

from flash.core.data.batch import _PostProcessor, _PreProcessor, _Sequential, default_uncollate


def test_sequential_str():
    sequential = _Sequential(
        Mock(name="preprocess"),
        torch.softmax,
        torch.as_tensor,
        torch.relu,
        RunningStage.TRAINING,
        True,
    )
    assert str(sequential) == (
        "_Sequential:\n"
        "\t(pre_tensor_transform): FuncModule(softmax)\n"
        "\t(to_tensor_transform): FuncModule(as_tensor)\n"
        "\t(post_tensor_transform): FuncModule(relu)\n"
        "\t(assert_contains_tensor): True\n"
        "\t(stage): RunningStage.TRAINING"
    )


def test_preprocessor_str():
    preprocessor = _PreProcessor(
        Mock(name="preprocess"),
        default_collate,
        torch.relu,
        torch.softmax,
        RunningStage.TRAINING,
        False,
        True,
    )
    assert str(preprocessor) == (
        "_PreProcessor:\n"
        "\t(per_sample_transform): FuncModule(relu)\n"
        "\t(collate_fn): FuncModule(default_collate)\n"
        "\t(per_batch_transform): FuncModule(softmax)\n"
        "\t(apply_per_sample_transform): False\n"
        "\t(on_device): True\n"
        "\t(stage): RunningStage.TRAINING"
    )


def test_postprocessor_str():
    postprocessor = _PostProcessor(
        default_uncollate,
        torch.relu,
        torch.softmax,
        None,
    )
    assert str(postprocessor) == (
        "_PostProcessor:\n"
        "\t(per_batch_transform): FuncModule(relu)\n"
        "\t(uncollate_fn): FuncModule(default_uncollate)\n"
        "\t(per_sample_transform): FuncModule(softmax)\n"
        "\t(serializer): None"
    )


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

    def test_named_tuple(self):
        B = 3  # batch_size

        Batch = namedtuple("Batch", ["x", "y"])
        batch = Batch(x=torch.rand(B, 4), y=torch.rand(B))

        output = default_uncollate(batch)
        assert isinstance(output, list)
        assert len(output) == B

        for sample in output:
            assert isinstance(sample, Batch)
            assert isinstance(sample.x, list)
            assert len(sample.x) == 4
            assert isinstance(sample.y, torch.Tensor)
            assert len(sample.y.shape) == 0
