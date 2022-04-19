from typing import Any, List, Mapping

from torch.utils.data._utils.collate import default_collate as torch_default_collate

from flash.core.data.io.input import DataKeys


def default_collate(batch: List[Any]) -> Any:
    """The :func:`flash.data.utilities.collate.default_collate` extends `torch.utils.data._utils.default_collate`
    to first extract any metadata from the samples in the batch (in the ``"metadata"`` key). The list of metadata
    entries will then be inserted into the collated result.

    Args:
        batch: The list of samples to collate.

    Returns:
        The collated batch.
    """
    metadata = [sample.pop(DataKeys.METADATA, None) if isinstance(sample, Mapping) else None for sample in batch]
    metadata = metadata if any(m is not None for m in metadata) else None

    collated_batch = torch_default_collate(batch)
    if metadata and isinstance(collated_batch, dict):
        collated_batch[DataKeys.METADATA] = metadata
    return collated_batch
