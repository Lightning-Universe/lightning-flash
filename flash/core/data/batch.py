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
from typing import Any, Sequence, TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from flash.core.data.io.input import ServeInput


class _ServeInputProcessor(torch.nn.Module):
    def __init__(
        self,
        serve_input: "ServeInput",
    ):
        super().__init__()
        self.serve_input = serve_input
        self.dataloader_collate_fn = self.serve_input._create_dataloader_collate_fn([])

    def forward(self, sample: str):
        sample = self.serve_input._call_load_sample(sample)
        sample = self.dataloader_collate_fn(sample)
        return sample


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
