from typing import Any, List, Optional

import numpy as np
from torch.utils.data import Dataset

from flash.core.data.properties import Properties
from flash.core.utilities.stages import RunningStage


class SplitDataset(Properties, Dataset):
    """SplitDataset is used to create Dataset Subset using indices.

    Args:

        dataset: A dataset to be split
        indices: List of indices to expose from the dataset
        use_duplicated_indices: Whether to allow duplicated indices.

    Example::

        split_ds = SplitDataset(dataset, indices=[10, 14, 25])

        split_ds = SplitDataset(dataset, indices=[10, 10, 10, 14, 25], use_duplicated_indices=True)
    """

    def __init__(
        self,
        dataset: Any,
        indices: List[int],
        running_stage: Optional[RunningStage] = None,
        use_duplicated_indices: bool = False,
    ) -> None:
        kwargs = {}
        if running_stage is not None:
            kwargs = dict(running_stage=running_stage)
        elif isinstance(dataset, Properties):
            kwargs = dict(running_stage=dataset._running_stage)
        super().__init__(**kwargs)

        if not isinstance(indices, list):
            raise TypeError("indices should be a list")

        if use_duplicated_indices:
            indices = list(indices)
        else:
            indices = list(np.unique(indices))

        if np.max(indices) >= len(dataset) or np.min(indices) < 0:
            raise ValueError(f"`indices` should be within [0, {len(dataset) -1}].")

        self.dataset = dataset
        self.indices = indices

    def __getattr__(self, key: str):
        if key != "dataset":
            return getattr(self.dataset, key)
        raise AttributeError

    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.indices[index]]

    def __len__(self) -> int:
        return len(self.indices)
