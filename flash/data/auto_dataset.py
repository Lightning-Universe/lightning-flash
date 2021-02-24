from typing import Any, Callable

import torch


class AutoDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data: Any,
        load_data: Callable,
        load_sample: Callable,
    ) -> None:
        super().__init__()

        self.data = data
        self.load_sample = load_sample
        self.load_data = load_data
        self._processed_data = self.load_data(self.data)

    def __getitem__(self, index: int) -> Any:
        return self.load_sample(self._processed_data[index])

    def __len__(self) -> int:
        return len(self._processed_data)
