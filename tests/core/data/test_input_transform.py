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

from flash.core.data.io.input_transform import Compose, InputTransform, LambdaInputTransform
from flash.core.data.transforms import ApplyToKeys
from flash.core.data.utilities.collate import default_collate
from flash.core.utilities.imports import _CORE_TESTING
from flash.core.utilities.stages import RunningStage


@pytest.mark.skipif(not _CORE_TESTING)
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
        transform = MyTransform()
        transform._populate_transforms_for_stage(RunningStage.TRAINING)

    class MyTransform(InputTransform):
        def input_per_batch_transform(self) -> Callable:
            return None

    with pytest.raises(MisconfigurationException, match="The hook input_per_batch_transform should return a function."):
        transform = MyTransform()
        transform._populate_transforms_for_stage(RunningStage.TRAINING)

    class MyTransform(InputTransform):
        def target_per_batch_transform(self) -> Callable:
            return super().target_per_batch_transform()

        def input_per_batch_transform(self) -> Callable:
            return fn

    transform = MyTransform()
    for stage in [RunningStage.TRAINING, RunningStage.VALIDATING, RunningStage.TESTING, RunningStage.PREDICTING]:
        transform._populate_transforms_for_stage(stage)
        transforms = transform._transform[stage].transforms
        assert list(transforms.keys()) == ["per_batch_transform", "collate"]
        assert isinstance(transforms["per_batch_transform"], ApplyToKeys)
        assert transforms["per_batch_transform"].keys == ["input"]
        assert transforms["collate"] == default_collate

    class MyTransform(InputTransform):
        def train_per_batch_transform(self) -> Callable:
            return self.train_per_batch_transform

        def target_per_batch_transform(self) -> Callable:
            return self.target_per_batch_transform

        def input_per_batch_transform(self) -> Callable:
            return self.input_per_batch_transform

    transform = MyTransform()

    # Tests for RunningStage.TRAINING
    transform._populate_transforms_for_stage(RunningStage.TRAINING)
    train_transforms = transform._transform[RunningStage.TRAINING].transforms
    assert list(train_transforms.keys()) == ["per_batch_transform", "collate"]
    assert train_transforms["per_batch_transform"] == transform.train_per_batch_transform

    # Tests for RunningStage.VALIDATING
    transform._populate_transforms_for_stage(RunningStage.VALIDATING)
    val_transforms = transform._transform[RunningStage.VALIDATING].transforms
    assert isinstance(val_transforms["per_batch_transform"], Compose)
    assert len(val_transforms["per_batch_transform"].transforms) == 2

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
        transform = MyTransform()
        transform._populate_transforms_for_stage(RunningStage.TRAINING)

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

    transform = MyTransform()

    # Tests for RunningStage.TRAINING
    transform._populate_transforms_for_stage(RunningStage.TRAINING)
    train_transforms = transform._transform[RunningStage.TRAINING].transforms
    assert list(train_transforms.keys()) == ["per_batch_transform", "collate"]
    assert isinstance(train_transforms["per_batch_transform"], Compose)
    assert len(train_transforms["per_batch_transform"].transforms) == 2
    assert train_transforms["collate"] == transform.train_collate

    # Tests for RunningStage.VALIDATING
    transform._populate_transforms_for_stage(RunningStage.VALIDATING)
    val_transforms = transform._transform[RunningStage.VALIDATING].transforms
    assert list(val_transforms.keys()) == ["per_batch_transform", "collate"]
    assert val_transforms["per_batch_transform"] == transform.train_per_batch_transform
    assert val_transforms["collate"] == transform.collate

    transform = LambdaInputTransform(transform=fn)
    for stage in [RunningStage.TRAINING, RunningStage.VALIDATING, RunningStage.TESTING, RunningStage.PREDICTING]:
        transform._populate_transforms_for_stage(stage)
        transforms = transform._transform[stage].transforms
        assert list(transforms.keys()) == ["per_sample_transform", "collate"]
        assert transforms["per_sample_transform"] == fn

    class MyTransform(InputTransform):
        def __init__(self, value: int):
            super().__init__()
            self.value = value

        def input_per_batch_transform(self) -> Callable:
            if self.value > 0:
                return self.input_per_batch_transform
            return super().input_per_batch_transform

    with pytest.raises(AttributeError, match="__init__"):
        MyTransform(1)

    class MyTransform(InputTransform):
        def __init__(self, value: int):
            self.value = value
            super().__init__()

        def input_per_batch_transform(self) -> Callable:
            if self.value > 0:
                return self.input_per_batch_transform
            return super().input_per_batch_transform

    MyTransform(1)


class CustomInputTransform(InputTransform):
    @staticmethod
    def custom_transform(x):
        return x

    def train_per_sample_transform(self):
        return self.custom_transform

    def train_per_batch_transform_on_device(self, *_, **__):
        return self.custom_transform

    def test_per_sample_transform(self, *_, **__):
        return self.custom_transform

    def test_per_batch_transform(self, *_, **__):
        return self.custom_transform

    def test_per_sample_transform_on_device(self, *_, **__):
        return self.custom_transform

    def test_per_batch_transform_on_device(self, *_, **__):
        return self.custom_transform

    def val_per_batch_transform(self, *_, **__):
        return self.custom_transform

    def val_per_sample_transform_on_device(self, *_, **__):
        return self.custom_transform

    def predict_per_sample_transform(self, *_, **__):
        return self.custom_transform

    def predict_per_sample_transform_on_device(self, *_, **__):
        return self.custom_transform

    def predict_per_batch_transform_on_device(self, *_, **__):
        return self.custom_transform


@pytest.mark.skipif(not _CORE_TESTING)
def test_check_transforms():

    input_transform = CustomInputTransform

    with pytest.raises(MisconfigurationException, match="are mutually exclusive"):
        input_transform()
