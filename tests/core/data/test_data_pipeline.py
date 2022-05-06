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
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from flash.core.data.data_pipeline import DataPipeline
from flash.core.data.io.input_transform import InputTransform
from flash.core.utilities.imports import _CORE_TESTING
from flash.core.utilities.stages import RunningStage


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
def test_is_overridden_recursive(tmpdir):
    class TestInputTransform(InputTransform):
        @staticmethod
        def custom_transform(x):
            return x

        def collate(self):
            return self.custom_transform

        def val_collate(self):
            return self.custom_transform

    input_transform = TestInputTransform(RunningStage.TRAINING)
    assert DataPipeline._is_overridden_recursive("collate", input_transform, InputTransform, prefix="val")
    assert DataPipeline._is_overridden_recursive("collate", input_transform, InputTransform, prefix="train")
    assert not DataPipeline._is_overridden_recursive(
        "per_batch_transform_on_device", input_transform, InputTransform, prefix="train"
    )
    assert not DataPipeline._is_overridden_recursive("per_batch_transform_on_device", input_transform, InputTransform)
    with pytest.raises(MisconfigurationException, match="This function doesn't belong to the parent class"):
        assert not DataPipeline._is_overridden_recursive("chocolate", input_transform, InputTransform)
