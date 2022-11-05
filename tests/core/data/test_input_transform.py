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

from flash.core.data.io.input_transform import InputTransform
from flash.core.utilities.imports import _CORE_TESTING
from flash.core.utilities.stages import RunningStage


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
def test_input_transform():
    def fn(x):
        return x + 1

    class MyTransform(InputTransform):
        def train_per_batch_transform(self) -> Callable:
            return self.train_per_batch_transform

    transform = MyTransform()

    # Tests for RunningStage.TRAINING
    transform._populate_transforms_for_stage(RunningStage.TRAINING)
    train_transforms = transform._transform[RunningStage.TRAINING].transforms
    assert list(train_transforms.keys()) == ["per_batch_transform", "collate"]
    assert train_transforms["per_batch_transform"] == transform.train_per_batch_transform

    class MyTransform(InputTransform):
        def per_batch_transform(self) -> Callable:
            return self.per_batch_transform

        def train_collate(self) -> Callable:
            return self.train_collate

        def collate(self) -> Callable:
            return self.collate

    transform = MyTransform()

    # Tests for RunningStage.TRAINING
    transform._populate_transforms_for_stage(RunningStage.TRAINING)
    train_transforms = transform._transform[RunningStage.TRAINING].transforms
    assert list(train_transforms.keys()) == ["per_batch_transform", "collate"]
    assert train_transforms["per_batch_transform"] == transform.per_batch_transform
    assert train_transforms["collate"] == transform.train_collate

    # Tests for RunningStage.VALIDATING
    transform._populate_transforms_for_stage(RunningStage.VALIDATING)
    val_transforms = transform._transform[RunningStage.VALIDATING].transforms
    assert list(val_transforms.keys()) == ["per_batch_transform", "collate"]
    assert val_transforms["per_batch_transform"] == transform.per_batch_transform
    assert val_transforms["collate"] == transform.collate

    class MyTransform(InputTransform):
        def __init__(self, value: int):
            super().__init__()
            self.value = value

        def per_batch_transform(self) -> Callable:
            if self.value > 0:
                return self.per_batch_transform
            return super().per_batch_transform

    with pytest.raises(AttributeError, match="__init__"):
        MyTransform(1)

    class MyTransform(InputTransform):
        def __init__(self, value: int):
            self.value = value
            super().__init__()

        def per_batch_transform(self) -> Callable:
            if self.value > 0:
                return self.per_batch_transform
            return super().per_batch_transform

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


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
def test_check_transforms():
    input_transform = CustomInputTransform

    with pytest.raises(TypeError, match="are mutually exclusive"):
        input_transform()
