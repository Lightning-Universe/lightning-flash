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
from typing import cast, Tuple

import pytest
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor

from flash.core.data.data_pipeline import DataPipeline, DataPipelineState
from flash.core.data.io.input import Input
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.io.output import Output
from flash.core.data.io.output_transform import OutputTransform
from flash.core.data.process import Deserializer
from flash.core.data.properties import ProcessState
from flash.core.utilities.stages import RunningStage


class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return torch.rand(1), torch.rand(1)

    def __len__(self) -> int:
        return 5


class TestDataPipelineState:
    @staticmethod
    def test_str():
        state = DataPipelineState()
        state.set_state(ProcessState())

        assert str(state) == (
            "DataPipelineState(state={<class 'flash.core.data.properties.ProcessState'>: ProcessState()})"
        )

    @staticmethod
    def test_get_state():
        state = DataPipelineState()
        assert state.get_state(ProcessState) is None


def test_data_pipeline_str():
    data_pipeline = DataPipeline(
        input=cast(Input, "input"),
        input_transform=cast(InputTransform, "input_transform"),
        output_transform=cast(OutputTransform, "output_transform"),
        output=cast(Output, "output"),
        deserializer=cast(Deserializer, "deserializer"),
    )

    expected = "input=input, deserializer=deserializer, "
    expected += "input_transform=input_transform, output_transform=output_transform, output=output"
    assert str(data_pipeline) == (f"DataPipeline({expected})")


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
