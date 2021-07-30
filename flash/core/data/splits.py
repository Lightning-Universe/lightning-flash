from typing import Any, List

import numpy as np
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import Dataset


class SplitDataset(Dataset):
    """SplitDataset is used to create Dataset Subset using indices.

    Args:

        dataset: A dataset to be splitted
        indices: List of indices to expose from the dataset
        use_duplicated_indices: Whether to allow duplicated indices.

    Example::

        split_ds = SplitDataset(dataset, indices=[10, 14, 25])

        split_ds = SplitDataset(dataset, indices=[10, 10, 10, 14, 25], use_duplicated_indices=True)
    """

    _INTERNAL_KEYS = ("dataset", "indices", "data")

    def __init__(self, dataset: Any, indices: List[int] = None, use_duplicated_indices: bool = False) -> None:
        if indices is None:
            indices = []
        if not isinstance(indices, list):
            raise MisconfigurationException("indices should be a list")

        if use_duplicated_indices:
            indices = list(indices)
        else:
            indices = list(np.unique(indices))

        if np.max(indices) >= len(dataset) or np.min(indices) < 0:
            raise MisconfigurationException(f"`indices` should be within [0, {len(dataset) -1}].")

        self.dataset = dataset
        self.indices = indices

    def __getattr__(self, key: str):
        if key not in self._INTERNAL_KEYS:
            return self.dataset.__getattribute__(key)
        raise AttributeError

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._INTERNAL_KEYS:
            self.__dict__[name] = value
        else:
            setattr(self.dataset, name, value)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.indices[index]]

    def __len__(self) -> int:
        return len(self.indices)
