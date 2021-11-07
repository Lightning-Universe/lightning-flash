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
import os
from typing import Any, Optional

import torch

from flash.core.data.batch import default_uncollate
from flash.core.data.properties import Properties


class OutputTransform(Properties):
    """The :class:`~flash.core.data.io.output_transform.OutputTransform` encapsulates all the data processing logic
    that should run after the model."""

    def __init__(self, save_path: Optional[str] = None):
        super().__init__()
        self._saved_samples = 0
        self._save_path = save_path

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
    def save_data(data: Any, path: str) -> None:
        """Saves all data together to a single path."""
        torch.save(data, path)

    @staticmethod
    def save_sample(sample: Any, path: str) -> None:
        """Saves each sample individually to a given path."""
        torch.save(sample, path)

    # TODO: Are those needed ?
    def format_sample_save_path(self, path: str) -> str:
        path = os.path.join(path, f"sample_{self._saved_samples}.ptl")
        self._saved_samples += 1
        return path

    def _save_data(self, data: Any) -> None:
        self.save_data(data, self._save_path)

    def _save_sample(self, sample: Any) -> None:
        self.save_sample(sample, self.format_sample_save_path(self._save_path))
