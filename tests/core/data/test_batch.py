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

from flash.core.data.batch import _Postprocessor, _Preprocessor, _Sequential, default_uncollate


def test_sequential_str():
    sequential = _Sequential(
        Mock(name="preprocess"),
        torch.softmax,
        torch.as_tensor,
        torch.relu,
        RunningStage.TRAINING,
        True,
    )
    assert "(pre_tensor_transform): <built-in method softmax" in str(sequential)
    assert "(to_tensor_transform): <built-in method as_tensor" in str(sequential)
    assert "(post_tensor_transform): <built-in method relu" in str(sequential)


def test_preprocessor_str():
    preprocessor = _Preprocessor(
        Mock(name="preprocess"),
        default_collate,
        torch.relu,
        torch.softmax,
        RunningStage.TRAINING,
        False,
        True,
    )
    assert "function default_collate" in str(preprocessor)
    assert "built-in method relu" in str(preprocessor)
    assert "built-in method softmax" in str(preprocessor)


def test_postprocessor_str():
    postprocessor = _Postprocessor(
        default_uncollate,
        torch.relu,
        torch.softmax,
        None,
    )
    breakpoint()
    assert "(uncollate_fn): <function default_uncollate" in str(postprocessor)
    assert "(per_batch_transform): <built-in method relu" in str(postprocessor)
    assert "(per_sample_transform): <built-in method softmax" in str(postprocessor)


class TestDefaultUncollate:

    BATCH_SIZE = 3

    @staticmethod
    def test_smoke():
        batch = torch.rand(2, 1)
        assert default_uncollate(batch) is not None

    @staticmethod
    def test_tensor_zero():
        batch = torch.tensor(1)
        output = default_uncollate(batch)
        assert_allclose(batch, output)

    @staticmethod
    def test_tensor_batch():
        batch = torch.rand(2, 1)
        output = default_uncollate(batch)
        assert isinstance(output, list)
        assert all(isinstance(x, torch.Tensor) for x in output)

    def test_sequence(self):
        batch = {
            "a": torch.rand(self.BATCH_SIZE, 4),
            "b": torch.rand(self.BATCH_SIZE, 2),
            "c": torch.rand(self.BATCH_SIZE),
        }

        output = default_uncollate(batch)
        assert isinstance(output, list)
        assert len(batch) == self.BATCH_SIZE

        for sample in output:
            assert list(sample.keys()) == ["a", "b", "c"]
            assert isinstance(sample["a"], torch.Tensor)
            assert len(sample["a"]) == 4
            assert isinstance(sample["b"], torch.Tensor)
            assert len(sample["b"]) == 2
            assert isinstance(sample["c"], torch.Tensor)
            assert len(sample["c"].shape) == 0

    def test_named_tuple(self):
        Batch = namedtuple("Batch", ["x", "y"])
        batch = Batch(x=torch.rand(self.BATCH_SIZE, 4), y=torch.rand(self.BATCH_SIZE))

        output = default_uncollate(batch)
        assert isinstance(output, list)
        assert len(output) == self.BATCH_SIZE

        for sample in output:
            assert isinstance(sample, Batch)
            assert isinstance(sample.x, torch.Tensor)
            assert len(sample.x) == 4
            assert isinstance(sample.y, torch.Tensor)
            assert len(sample.y.shape) == 0
