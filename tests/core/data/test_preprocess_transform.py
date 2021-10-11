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

from flash.core.data.preprocess_transform import PreTransform, PreTransformPlacement


def test_preprocess_transform():

    transform = PreTransform(running_stage=RunningStage.TRAINING)

    assert str(transform) == "PreTransform(running_stage=train, transform=None)"

    def fn():
        pass

    transform = PreTransform.from_transform(running_stage=RunningStage.TRAINING, transform=fn)
    assert transform.transform == {PreTransformPlacement.PER_SAMPLE_TRANSFORM: fn}

    class MyPreTransform(PreTransform):
        def configure_transforms(self) -> Optional[Dict[str, Callable]]:
            return None

    transform = MyPreTransform(running_stage=RunningStage.TRAINING)
    assert str(transform) == "MyPreTransform(running_stage=train, transform=None)"

    class MyPreTransform(PreTransform):
        def fn(self):
            pass

        def configure_transforms(self) -> Optional[Dict[str, Callable]]:
            return {PreTransformPlacement.PER_SAMPLE_TRANSFORM: self.fn if self.training else fn}

    transform = MyPreTransform(running_stage=RunningStage.TRAINING)
    assert transform.transform == {PreTransformPlacement.PER_SAMPLE_TRANSFORM: transform.fn}

    transform = MyPreTransform(running_stage=RunningStage.TESTING)
    assert transform.transform == {PreTransformPlacement.PER_SAMPLE_TRANSFORM: fn}

    class FailureMyPreTransform(PreTransform):
        def configure_transforms(self) -> Optional[Dict[str, Callable]]:
            return {"wrong_key": fn}

    with pytest.raises(MisconfigurationException, match="train_transform contains {'wrong_key'}"):
        transform = FailureMyPreTransform(running_stage=RunningStage.TRAINING)

    with pytest.raises(MisconfigurationException, match="test_transform contains {'wrong_key'}"):
        transform = FailureMyPreTransform(running_stage=RunningStage.TESTING)
