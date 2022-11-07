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
import functools
from typing import Any, Callable, List, Mapping

from torch.utils.data._utils.collate import default_collate as torch_default_collate

from flash.core.data.io.input import DataKeys


def _wrap_collate(collate: Callable, batch: List[Any]) -> Any:
    # Needed for learn2learn integration
    if len(batch) == 1 and isinstance(batch[0], list):
        batch = batch[0]

    metadata = [sample.pop(DataKeys.METADATA, None) if isinstance(sample, Mapping) else None for sample in batch]
    metadata = metadata if any(m is not None for m in metadata) else None

    collated_batch = collate(batch)

    if metadata and isinstance(collated_batch, dict):
        collated_batch[DataKeys.METADATA] = metadata
    return collated_batch


def wrap_collate(collate):
    """:func:`flash.data.utilities.collate.wrap_collate` is a utility that can be used to wrap an existing collate
    function to handle the metadata separately from the rest of the batch (giving a list of the metadata from the
    samples in the output).

    Args:
        collate: The collate function to wrap.

    Returns:
        The wrapped collate function.
    """
    return functools.partial(_wrap_collate, collate)


_default_collate = wrap_collate(torch_default_collate)


def default_collate(batch: List[Any]) -> Any:
    """The :func:`flash.data.utilities.collate.default_collate` extends `torch.utils.data._utils.default_collate`
    to first extract any metadata from the samples in the batch (in the ``"metadata"`` key). The list of metadata
    entries will then be inserted into the collated result.

    Args:
        batch: The list of samples to collate.

    Returns:
        The collated batch.
    """
    return _default_collate(batch)
