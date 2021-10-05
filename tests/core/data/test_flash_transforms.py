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
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from flash.core.data.flash_transform import FlashTransform, TransformPlacement


def test_flash_transform():

    assert not FlashTransform.from_transform(None, RunningStage.TRAINING, None)

    def fn():
        pass

    flash_transform = FlashTransform.from_transform(fn, RunningStage.TRAINING, None)
    assert flash_transform.running_stage == RunningStage.TRAINING
    assert flash_transform.transform == {TransformPlacement.PER_SAMPLE_TRANSFORM: fn}

    class TestTransform(FlashTransform):
        pass

    transform = TestTransform(running_stage=RunningStage.TRAINING, transform=None)
    assert not transform.transform

    class TestTransform(FlashTransform):
        def configure_transforms(self):
            return {"something": None}

    with pytest.raises(MisconfigurationException, match="train_transform contains {'something'}"):
        transform = TestTransform(running_stage=RunningStage.TRAINING, transform=None)
        assert not transform.transform

    class TestTransform(FlashTransform):
        def configure_transforms(self):
            return {TransformPlacement.PER_SAMPLE_TRANSFORM: fn}

    transform = TestTransform(RunningStage.TRAINING)
    assert transform.transform == {TransformPlacement.PER_SAMPLE_TRANSFORM: fn}
