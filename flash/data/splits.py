from typing import Any, List

import numpy as np
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import Dataset


class SplitDataset(Dataset):

    def __init__(self, dataset: Any, indices: List[int] = [], use_duplicated_indices: bool = False) -> None:
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

    def __getitem__(self, index) -> Any:
        return self.dataset[self.indices[index]]

    def __len__(self) -> int:
        return len(self.indices)
