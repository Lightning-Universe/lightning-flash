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
from typing import Any, Callable, Optional, Sequence

import torch
from torch import Tensor

from flash.core.data.batch import default_uncollate
from flash.core.data.properties import Properties
from flash.core.data.utils import convert_to_modules


class OutputTransform(Properties):
    """The :class:`~flash.core.data.io.output_transform.OutputTransform` encapsulates all the data processing logic
    that should run after the model."""

    @staticmethod
    def per_batch_transform(batch: Any) -> Any:
        """Transforms to apply on a whole batch before uncollation to individual samples.

        Can involve both CPU and Device transforms as this is not applied in separate workers.
        """
        return batch

    @staticmethod
    def per_sample_transform(sample: Any) -> Any:
        """Transforms to apply to a single sample after splitting up the batch.

        Can involve both CPU and Device transforms as this is not applied in separate workers.
        """
        return sample

    @staticmethod
    def uncollate(batch: Any) -> Any:
        """Uncollates a batch into single samples.

        Tries to preserve the type whereever possible.
        """
        return default_uncollate(batch)


class _OutputTransformProcessor(torch.nn.Module):
    """This class is used to encapsultate the following functions of a OutputTransform Object:

    Inside main process:
        per_batch_transform: Function to transform a batch
        per_sample_transform: Function to transform an individual sample
        uncollate_fn: Function to split a batch into samples
        per_sample_transform: Function to transform an individual sample
        is_serving: Whether the Postprocessor is used in serving mode.
    """

    def __init__(
        self,
        uncollate_fn: Callable,
        per_batch_transform: Callable,
        per_sample_transform: Callable,
        output: Optional[Callable],
        is_serving: bool = False,
    ):
        super().__init__()
        self.uncollate_fn = convert_to_modules(uncollate_fn)
        self.per_batch_transform = convert_to_modules(per_batch_transform)
        self.per_sample_transform = convert_to_modules(per_sample_transform)
        self.output = convert_to_modules(output)
        self.is_serving = is_serving

    def forward(self, batch: Sequence[Any]):
        if batch is None:
            return batch

        uncollated = self.uncollate_fn(self.per_batch_transform(batch))

        final_preds = [self.per_sample_transform(sample) for sample in uncollated]

        if self.output is not None:
            final_preds = [self.output(sample) for sample in final_preds]

        if isinstance(uncollated, Tensor) and isinstance(final_preds[0], Tensor):
            return torch.stack(final_preds)
        return type(final_preds)(final_preds)

    def __str__(self) -> str:
        return (
            "_OutputTransformProcessor:\n"
            f"\t(per_batch_transform): {str(self.per_batch_transform)}\n"
            f"\t(uncollate_fn): {str(self.uncollate_fn)}\n"
            f"\t(per_sample_transform): {str(self.per_sample_transform)}\n"
            f"\t(output): {str(self.output)}"
        )
