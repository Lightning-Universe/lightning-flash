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
from typing import Callable, Dict, Optional

import pytest
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data.dataloader import default_collate

from flash.core.data.preprocess_transform import PreTransform, PreTransformPlacement
from flash.core.registry import FlashRegistry


def test_preprocess_transform():

    transform = PreTransform(running_stage=RunningStage.TRAINING)

    def fn():
        pass

    transform = PreTransform.from_transform(running_stage=RunningStage.TRAINING, transform=fn)
    assert transform.transform == {PreTransformPlacement.PER_SAMPLE_TRANSFORM: fn}

    class MyPreTransform(PreTransform):
        def configure_transforms(self) -> Optional[Dict[str, Callable]]:
            return None

    transform = MyPreTransform(running_stage=RunningStage.TRAINING)
    assert "PreTransform(running_stage=train, transform={<PreTransformPlacement.COLLATE: 'collate'>" in str(transform)

    class MyPreTransform(PreTransform):
        def fn(self):
            pass

        def configure_per_sample_transform(self) -> Optional[Dict[str, Callable]]:
            return self.fn if self.training else fn

    transform = MyPreTransform(running_stage=RunningStage.TRAINING)
    assert transform.transform == {
        PreTransformPlacement.PER_SAMPLE_TRANSFORM: transform.fn,
        PreTransformPlacement.COLLATE: default_collate,
    }

    transform = MyPreTransform(running_stage=RunningStage.TESTING)
    assert transform.transform == {
        PreTransformPlacement.PER_SAMPLE_TRANSFORM: fn,
        PreTransformPlacement.COLLATE: default_collate,
    }

    transform_registry = FlashRegistry("transforms")
    transform_registry(fn=MyPreTransform, name="something")

    transform = PreTransform.from_transform(
        running_stage=RunningStage.TRAINING, transform="something", transform_registry=transform_registry
    )

    assert isinstance(transform, MyPreTransform)
    assert transform.transform == {
        PreTransformPlacement.PER_SAMPLE_TRANSFORM: transform.fn,
        PreTransformPlacement.COLLATE: default_collate,
    }

    collate_fn = transform.dataloader_collate_fn
    assert collate_fn.collate_fn.func == transform.collate
    assert collate_fn.per_sample_transform.func == transform.per_sample_transform
    assert collate_fn.per_batch_transform.func == transform.per_batch_transform

    on_after_batch_transfer_fn = transform.on_after_batch_transfer_fn
    assert on_after_batch_transfer_fn.collate_fn.func == transform._identity
    assert on_after_batch_transfer_fn.per_sample_transform.func == transform.per_sample_transform_on_device
    assert on_after_batch_transfer_fn.per_batch_transform.func == transform.per_batch_transform_on_device

    class MyPreTransform(PreTransform):
        def configure_per_batch_transform(self, *args, **kwargs) -> Callable:
            return fn

        def configure_per_sample_transform_on_device(self, *args, **kwargs) -> Callable:
            return fn

    with pytest.raises(MisconfigurationException, match="`per_batch_transform` and `per_sample_transform_on_device`"):
        transform = MyPreTransform(running_stage=RunningStage.TESTING)
