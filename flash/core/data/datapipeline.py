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

from typing import Any

from torch import Tensor
from torch.utils.data._utils.collate import default_collate


class DataPipeline:
    """
    This class purpose is to facilitate the conversion of raw data to processed or batched data and back.
    Several hooks are provided for maximum flexibility.

    Example::

        .. code-block:: python

            class MyTextDataPipeline(DataPipeline):
                def __init__(self, tokenizer, padder):
                    self.tokenizer = tokenizer
                    self.padder = padder

                def before_collate(self, samples):
                    # encode each input sequence
                    return [self.tokenizer.encode(sample) for sample in samplers]

                def after_collate(self, batch):
                    # pad tensor elements to the maximum length in the batch
                    return self.padder(batch)

                def after_uncollate(self, samples):
                    # decode each input sequence
                    return [self.tokenizer.decode(sample) for sample in samples]

    """

    def before_collate(self, samples: Any) -> Any:
        """Override to apply transformations to samples"""
        return samples

    def collate(self, samples: Any) -> Any:
        """Override to convert a set of samples to a batch"""
        if not isinstance(samples, Tensor):
            return default_collate(samples)
        return samples

    def after_collate(self, batch: Any) -> Any:
        """Override to apply transformations to the batch"""
        return batch

    def collate_fn(self, samples: Any) -> Any:
        """
        Utility function to convert raw data to batched data

        ``collate_fn`` as used in ``torch.utils.data.DataLoader``.
        To avoid the before/after collate transformations, please use ``collate``.
        """
        samples = self.before_collate(samples)
        batch = self.collate(samples)
        batch = self.after_collate(batch)
        return batch

    def before_uncollate(self, batch: Any) -> Any:
        """Override to apply transformations to the batch"""
        return batch

    def uncollate(self, batch: Any) -> Any:
        """Override to convert a batch to a set of samples"""
        samples = batch
        return samples

    def after_uncollate(self, samples: Any) -> Any:
        """Override to apply transformations to samples"""
        return samples

    def uncollate_fn(self, batch: Any) -> Any:
        """Utility function to convert batched data back to raw data"""
        batch = self.before_uncollate(batch)
        samples = self.uncollate(batch)
        samples = self.after_uncollate(samples)
        return samples
