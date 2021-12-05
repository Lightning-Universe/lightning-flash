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
from typing import Callable

import pytest
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data.dataloader import default_collate

from flash.core.data.input_transform import Compose, InputTransform, LambdaInputTransform
from flash.core.data.transforms import ApplyToKeys
from flash.core.utilities.stages import RunningStage


def test_input_transform():
    def fn(x):
        return x + 1

    class MyTransform(InputTransform):
        def per_batch_transform(self) -> Callable:
            return super().per_batch_transform()

        def input_per_batch_transform(self) -> Callable:
            return fn

    with pytest.raises(
        MisconfigurationException,
        match="Only one of per_batch_transform or input_per_batch_transform can be overridden",
    ):
        MyTransform(running_stage=RunningStage.TRAINING)

    class MyTransform(InputTransform):
        def input_per_batch_transform(self) -> Callable:
            return None

    with pytest.raises(MisconfigurationException, match="The hook input_per_batch_transform should return a function."):
        MyTransform(running_stage=RunningStage.TRAINING)

    class MyTransform(InputTransform):
        def target_per_batch_transform(self) -> Callable:
            return super().target_per_batch_transform()

        def input_per_batch_transform(self) -> Callable:
            return fn

    transform = MyTransform(running_stage=RunningStage.TRAINING)
    assert list(transform._transform.keys()) == ["per_batch_transform", "collate"]
    assert isinstance(transform._transform["per_batch_transform"], ApplyToKeys)
    assert transform._transform["per_batch_transform"].keys == ["input"]
    assert transform._transform["collate"] == default_collate

    class MyTransform(InputTransform):
        def train_per_batch_transform(self) -> Callable:
            return self.train_per_batch_transform

        def target_per_batch_transform(self) -> Callable:
            return self.target_per_batch_transform

        def input_per_batch_transform(self) -> Callable:
            return self.input_per_batch_transform

    transform = MyTransform(running_stage=RunningStage.TRAINING)
    assert list(transform._transform.keys()) == ["per_batch_transform", "collate"]
    assert transform._transform["per_batch_transform"] == transform.train_per_batch_transform

    transform = MyTransform(running_stage=RunningStage.VALIDATING)
    assert isinstance(transform._transform["per_batch_transform"], Compose)
    assert len(transform._transform["per_batch_transform"].transforms) == 2

    class MyTransform(InputTransform):
        def train_per_batch_transform(self) -> Callable:
            return self.train_per_batch_transform

        def train_target_per_batch_transform(self) -> Callable:
            return super().target_per_batch_transform()

        def input_per_batch_transform(self) -> Callable:
            return fn

    with pytest.raises(
        MisconfigurationException,
        match="Only one of train_per_batch_transform or train_target_per_batch_transform can be overridden.",
    ):
        MyTransform(running_stage=RunningStage.TRAINING)

    class MyTransform(InputTransform):
        def per_batch_transform(self) -> Callable:
            return self.train_per_batch_transform

        def train_target_per_batch_transform(self) -> Callable:
            return self.train_target_per_batch_transform

        def train_input_per_batch_transform(self) -> Callable:
            return fn

        def train_collate(self) -> Callable:
            return self.train_collate

        def collate(self) -> Callable:
            return self.collate

    transform = MyTransform(running_stage=RunningStage.TRAINING)
    assert list(transform._transform.keys()) == ["per_batch_transform", "collate"]
    assert isinstance(transform._transform["per_batch_transform"], Compose)
    assert len(transform._transform["per_batch_transform"].transforms) == 2
    assert transform._transform["collate"] == transform.train_collate

    transform = MyTransform(running_stage=RunningStage.VALIDATING)
    assert list(transform._transform.keys()) == ["per_batch_transform", "collate"]
    assert transform._transform["per_batch_transform"] == transform.train_per_batch_transform
    assert transform._transform["collate"] == transform.collate

    transform = LambdaInputTransform(RunningStage.TRAINING, transform=fn)
    assert list(transform._transform.keys()) == ["per_sample_transform", "collate"]
    assert transform._transform["per_sample_transform"] == fn

    class MyTransform(InputTransform):
        def __init__(self, value: int, running_stage: RunningStage):
            super().__init__(running_stage)
            self.value = value

        def input_per_batch_transform(self) -> Callable:
            if self.value > 0:
                return self.input_per_batch_transform
            return super().input_per_batch_transform

    with pytest.raises(AttributeError, match="__init__"):
        MyTransform(1, running_stage=RunningStage.TRAINING)

    class MyTransform(InputTransform):
        def __init__(self, value: int, running_stage: RunningStage):
            self.value = value
            super().__init__(running_stage)

        def input_per_batch_transform(self) -> Callable:
            if self.value > 0:
                return self.input_per_batch_transform
            return super().input_per_batch_transform

    MyTransform(1, running_stage=RunningStage.TRAINING)
