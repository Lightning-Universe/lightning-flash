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
from typing import Any, Sequence

from flash.core.data.batch import default_uncollate


class OutputTransform:
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

        Tries to preserve the type wherever possible.
        """
        return default_uncollate(batch)

    def __call__(self, batch: Sequence[Any]):
        if batch is None:
            return batch

        uncollated = self.uncollate(self.per_batch_transform(batch))

        return [self.per_sample_transform(sample) for sample in uncollated]
