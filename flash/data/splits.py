from typing import Any, List

from torch.utils.data import Dataset


class SplitDataset(Dataset):

    def __init__(self, dataset: Dataset, indices: List[int] = []) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index) -> Any:
        return self.dataset[self.indices[index]]

    def __len__(self) -> int:
        return len(self.indices)
