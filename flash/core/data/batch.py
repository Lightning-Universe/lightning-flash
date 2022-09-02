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
from typing import Any, Callable, List, TYPE_CHECKING

from torch import nn, Tensor

from flash.core.data.utilities.classification import _is_list_like

if TYPE_CHECKING:
    from flash.core.data.io.input import ServeInput


class _ServeInputProcessor(nn.Module):
    def __init__(
        self,
        serve_input: "ServeInput",
        collate_fn: Callable,
    ):
        super().__init__()
        self.serve_input = serve_input
        self.collate_fn = collate_fn

    def forward(self, sample: str):
        sample = self.serve_input._call_load_sample(sample)
        if not isinstance(sample, list):
            sample = [sample]
        sample = self.collate_fn(sample)
        return sample


def _is_list_like_excluding_str(x):
    return _is_list_like(x) and str(x) != x


def default_uncollate(batch: Any) -> List[Any]:
    """This function is used to uncollate a batch into samples. The following conditions are used:

    - if the ``batch`` is a ``dict``, the result will be a list of dicts
    - if the ``batch`` is list-like, the result is guaranteed to be a list

    Args:
        batch: The batch of outputs to be uncollated.

    Returns:
        The uncollated list of predictions.

    Raises:
        ValueError: If the input is a ``dict`` whose values are not all list-like.
        ValueError: If the input is a ``dict`` whose values are not all the same length.
        ValueError: If the input is not a ``dict`` or list-like.
    """
    if isinstance(batch, dict):
        if any(not _is_list_like_excluding_str(sub_batch) for sub_batch in batch.values()):
            raise ValueError("When uncollating a dict, all sub-batches (values) are expected to be list-like.")
        if len({len(sub_batch) for sub_batch in batch.values()}) > 1:
            raise ValueError("When uncollating a dict, all sub-batches (values) are expected to have the same length.")
        elements = [default_uncollate(element) for element in zip(*batch.values())]
        return [dict(zip(batch.keys(), element)) for element in elements]
    if isinstance(batch, (list, tuple, Tensor)):
        return list(batch)
    raise ValueError(
        "The batch of outputs to be uncollated is expected to be a `dict` or list-like "
        f"(e.g. `Tensor`, `list`, `tuple`, etc.), but got input of type: {type(batch)}"
    )
