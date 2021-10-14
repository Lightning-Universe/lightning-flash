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
from functools import partial
from typing import Callable, Dict, Optional

import pytest
import torch
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data.dataloader import default_collate

from flash.core.data.input_transform import InputTransform, InputTransformPlacement
from flash.core.registry import FlashRegistry


def test_input_transform():

    transform = InputTransform(running_stage=RunningStage.TRAINING)

    assert "InputTransform(running_stage=train, transform={<InputTransformPlacement.COLLATE: 'collate'>" in str(
        transform
    )

    def fn(x):
        return x + 1

    transform = InputTransform.from_train_transform(transform=fn)
    assert transform.transform == {InputTransformPlacement.PER_SAMPLE_TRANSFORM: fn}

    transform = InputTransform.from_val_transform(transform=fn)
    assert transform.transform == {InputTransformPlacement.PER_SAMPLE_TRANSFORM: fn}

    transform = InputTransform.from_test_transform(transform=fn)
    assert transform.transform == {InputTransformPlacement.PER_SAMPLE_TRANSFORM: fn}

    transform = InputTransform.from_predict_transform(transform=fn)
    assert transform.transform == {InputTransformPlacement.PER_SAMPLE_TRANSFORM: fn}

    class MyInputTransform(InputTransform):
        def configure_transforms(self) -> Optional[Dict[str, Callable]]:
            return None

    transform = MyInputTransform(running_stage=RunningStage.TRAINING)
    assert not transform._current_fn
    assert "InputTransform(running_stage=train, transform={<InputTransformPlacement.COLLATE: 'collate'>" in str(
        transform
    )

    class MyInputTransform(InputTransform):
        def fn(self, x):
            return x + 1

        def configure_per_sample_transform(self) -> Optional[Dict[str, Callable]]:
            return self.fn if self.training else fn

    transform = MyInputTransform(running_stage=RunningStage.TRAINING)
    assert transform.transform == {
        InputTransformPlacement.PER_SAMPLE_TRANSFORM: transform.fn,
        InputTransformPlacement.COLLATE: default_collate,
    }

    transform._current_fn = "per_sample_transform"
    assert transform.current_transform == transform.fn
    assert transform.per_sample_transform(1) == 2
    assert transform.per_sample_transform([1, 2]) == [2, 3]

    transform._current_fn = "per_sample_transform_on_device"
    assert transform.current_transform == transform._identity
    assert transform.per_sample_transform_on_device(1) == 1
    assert transform.per_sample_transform_on_device([1, 2]) == [1, 2]

    transform._current_fn = "collate"
    assert transform.current_transform == default_collate
    assert torch.equal(transform.collate([0, 1]), torch.tensor([0, 1]))

    transform._current_fn = "per_batch_transform"
    assert transform.current_transform == transform._identity
    assert transform.per_batch_transform(2) == 2

    transform = MyInputTransform(running_stage=RunningStage.TESTING)
    assert transform.transform == {
        InputTransformPlacement.PER_SAMPLE_TRANSFORM: fn,
        InputTransformPlacement.COLLATE: default_collate,
    }

    assert transform.transforms == {
        "transform": {
            InputTransformPlacement.PER_SAMPLE_TRANSFORM: fn,
            InputTransformPlacement.COLLATE: default_collate,
        }
    }

    input_transforms_registry = FlashRegistry("transforms")
    input_transforms_registry(fn=MyInputTransform, name="something")

    transform = InputTransform.from_transform(
        running_stage=RunningStage.TRAINING, transform="something", input_transforms_registry=input_transforms_registry
    )

    transform = transform.from_transform(
        running_stage=RunningStage.TRAINING, transform=transform, input_transforms_registry=input_transforms_registry
    )

    assert isinstance(transform, MyInputTransform)
    assert transform.transform == {
        InputTransformPlacement.PER_SAMPLE_TRANSFORM: transform.fn,
        InputTransformPlacement.COLLATE: default_collate,
    }

    collate_fn = transform.dataloader_collate_fn
    assert collate_fn.collate_fn.func == transform.collate
    assert collate_fn.per_sample_transform.func == transform.per_sample_transform
    assert collate_fn.per_batch_transform.func == transform.per_batch_transform

    on_after_batch_transfer_fn = transform.on_after_batch_transfer_fn
    assert on_after_batch_transfer_fn.collate_fn.func == transform._identity
    assert on_after_batch_transfer_fn.per_sample_transform.func == transform.per_sample_transform_on_device
    assert on_after_batch_transfer_fn.per_batch_transform.func == transform.per_batch_transform_on_device

    assert transform._collate_in_worker_from_transform

    class MyInputTransform(InputTransform):
        def configure_transforms(self) -> Optional[Dict[str, Callable]]:
            return {
                InputTransformPlacement.PER_SAMPLE_TRANSFORM_ON_DEVICE: fn,
            }

        def configure_per_batch_transform(self):
            return fn

    with pytest.raises(MisconfigurationException, match="`per_batch_transform` and `per_sample_transform_on_device`"):
        transform = MyInputTransform(running_stage=RunningStage.TESTING)

    with pytest.raises(MisconfigurationException, match="The format for the transform isn't correct"):
        InputTransform.from_transform(1, running_stage=RunningStage.TRAINING)

    class MyInputTransform(InputTransform):
        def configure_transforms(self) -> Optional[Dict[str, Callable]]:
            return {
                InputTransformPlacement.COLLATE: fn,
                InputTransformPlacement.PER_SAMPLE_TRANSFORM_ON_DEVICE: fn,
                InputTransformPlacement.PER_BATCH_TRANSFORM_ON_DEVICE: fn,
            }

    transform = MyInputTransform(running_stage=RunningStage.TESTING)
    assert not transform._collate_in_worker_from_transform

    def compose(x, funcs):
        for f in funcs:
            x = f(x)
        return x

    transform = InputTransform.from_transform(
        transform=partial(compose, funcs=[fn, fn]), running_stage=RunningStage.TRAINING
    )
    assert transform[InputTransformPlacement.PER_SAMPLE_TRANSFORM](1) == 3


def test_transform_with_registry():
    def fn():
        pass

    class MyInputTransform(InputTransform):
        def configure_transforms(self, name: str = "lightning") -> Optional[Dict[str, Callable]]:
            self.name = name
            return {
                InputTransformPlacement.PER_SAMPLE_TRANSFORM_ON_DEVICE: fn,
            }

    registry = FlashRegistry("transforms")
    registry(name="custom", fn=MyInputTransform)

    transform = InputTransform.from_train_transform(transform="custom", input_transforms_registry=registry)
    assert isinstance(transform, MyInputTransform)
    assert transform.name == "lightning"

    transform = InputTransform.from_train_transform(
        transform=("custom", {"name": "flash"}), input_transforms_registry=registry
    )
    assert isinstance(transform, MyInputTransform)
    assert transform.name == "flash"

    transform = InputTransform.from_train_transform(transform=None, input_transforms_registry=registry)
    assert transform is None

    transform = InputTransform.from_train_transform(transform=None, input_transforms_registry=registry)
    assert transform is None

    with pytest.raises(
        MisconfigurationException, match="The transform should be provided as a tuple with the following types"
    ):
        transform = InputTransform.from_train_transform(transform=("custom", None), input_transforms_registry=registry)

    with pytest.raises(MisconfigurationException, match="The format for the transform isn't correct"):
        transform = InputTransform.from_train_transform(transform=1, input_transforms_registry=registry)
