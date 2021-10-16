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
from typing import Any, Mapping, Optional, Sequence, Tuple

from pytorch_lightning.utilities.enums import LightningEnum

from flash.core.data.batch import default_uncollate
from flash.core.data.properties import Properties
from flash.core.data_v2.io.input import InputDataKeys


class OutputTransformPlacement(LightningEnum):
    PER_BATCH_TRANSFORM = "per_batch_transform"
    UNCOLLATE = "uncollate"
    PER_SAMPLE_TRANSFORM = "per_sample_transform"


class OutputTransform(Properties):
    """The :class:`~flash.core.data_v2.transforms.output_transform.OutputTransform` encapsulates all the data
    processing logic that should run after the model."""

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

    @staticmethod
    def _extract_metadata(batch: Any) -> Tuple[Any, Optional[Any]]:
        metadata = None
        if isinstance(batch, Mapping) and InputDataKeys.METADATA in batch:
            metadata = batch.pop(InputDataKeys.METADATA, None)
        return batch, metadata

    def __call__(self, batch: Sequence[Any]):
        batch, metadata = self._extract_metadata(batch)
        uncollated = self.uncollate(self.per_batch_transform(batch))
        if metadata:
            for sample, sample_metadata in zip(uncollated, metadata):
                sample[InputDataKeys.METADATA] = sample_metadata

        return [self.per_sample_transform(sample) for sample in uncollated]
