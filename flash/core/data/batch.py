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
from typing import Any, Callable, Sequence, TYPE_CHECKING

import torch
from torch import Tensor

from flash.core.data.callback import ControlFlow
from flash.core.data.utils import convert_to_modules, CurrentFuncContext, CurrentRunningStageContext
from flash.core.utilities.stages import RunningStage

if TYPE_CHECKING:
    from flash.core.data.io.input_transform import InputTransform
    from flash.core.data.process import Deserializer, Serializer


class _DeserializeProcessor(torch.nn.Module):
    def __init__(
        self,
        deserializer: "Deserializer",
        input_transform: "InputTransform",
        pre_tensor_transform: Callable,
        to_tensor_transform: Callable,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.callback = ControlFlow(self.input_transform.callbacks)
        self.deserializer = convert_to_modules(deserializer)
        self.pre_tensor_transform = convert_to_modules(pre_tensor_transform)
        self.to_tensor_transform = convert_to_modules(to_tensor_transform)

        self._current_stage_context = CurrentRunningStageContext(RunningStage.PREDICTING, input_transform, reset=False)
        self._pre_tensor_transform_context = CurrentFuncContext("pre_tensor_transform", input_transform)
        self._to_tensor_transform_context = CurrentFuncContext("to_tensor_transform", input_transform)

    def forward(self, sample: str):

        sample = self.deserializer(sample)

        with self._current_stage_context:
            with self._pre_tensor_transform_context:
                sample = self.pre_tensor_transform(sample)
                self.callback.on_pre_tensor_transform(sample, RunningStage.PREDICTING)

            with self._to_tensor_transform_context:
                sample = self.to_tensor_transform(sample)
                self.callback.on_to_tensor_transform(sample, RunningStage.PREDICTING)

        return sample


class _SerializeProcessor(torch.nn.Module):
    def __init__(
        self,
        serializer: "Serializer",
    ):
        super().__init__()
        self.serializer = convert_to_modules(serializer)

    def forward(self, sample):
        return self.serializer(sample)


def default_uncollate(batch: Any):
    """
    This function is used to uncollate a batch into samples.
    Examples:
        >>> a, b = default_uncollate(torch.rand((2,1)))
    """

    batch_type = type(batch)

    if isinstance(batch, Tensor):
        if len(batch.shape) == 0:  # 0 shape tensors
            return batch
        return list(torch.unbind(batch, 0))

    if isinstance(batch, dict):
        return [batch_type(dict(zip(batch, default_uncollate(t)))) for t in zip(*batch.values())]

    if isinstance(batch, tuple) and hasattr(batch, "_fields"):  # namedtuple
        return [batch_type(*sample) for sample in zip(*batch)]

    if isinstance(batch, Sequence) and not isinstance(batch, str):
        return [sample for sample in batch]

    return batch
