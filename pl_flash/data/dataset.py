from typing import Mapping, Sequence
from torch.utils.data import Dataset


class PredefinedSequenceDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        assert isinstance(data, Sequence)
        # check all elements of sequence have same length
        lengths = list(set([len(tmp) for tmp in data]))
        assert len(lengths) == 1
        self.data = data
        self.length = lengths[0]

    def __getitem__(self, index: int) -> tuple:
        return tuple([tmp[index] for tmp in self.data])

    def __len__(self) -> int:
        return self.length


class PredefinedMappingDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        assert isinstance(data, Mapping)
        # check all elements of dict have same length
        lengths = list(set([len(tmp) for tmp in data.values()]))
        assert len(lengths) == 1
        self.data = data
        self.length = lengths[0]

    def __getitem__(self, index: int) -> dict:
        return {k: v[index] for k, v in self.data.items()}

    def __len__(self) -> int:
        return self.length

