from typing import Mapping, Sequence
from torch.utils.data import Dataset


class PredefinedSequenceDataset(Dataset):
    """A Dataset of predefined sequences.

    Args:
        data: should be a sequence of sequence of samples
    """

    def __init__(self, data: Sequence) -> None:
        super().__init__()
        assert isinstance(data, Sequence)
        # check all elements of sequence have same length
        lengths = list(set([len(tmp) for tmp in data]))
        assert len(lengths) == 1
        self.data = data
        self.length = lengths[0]

    def __getitem__(self, index: int) -> tuple:
        """Returns the corresponding item for each part of the sample

        Args:
            index: the integer specifying the actual sample

        Returns:
            tuple: the sample corresponding to :attr:`index`
        """
        return tuple([tmp[index] for tmp in self.data])

    def __len__(self) -> int:
        """Returns the number of samples in this dataset

        Returns:
            int: the dataset length
        """
        return self.length


class PredefinedMappingDataset(Dataset):
    """A Dataset of predefined mappings.

    Args:
        data: should be a mapping of sequence of samples
    """

    def __init__(self, data) -> None:
        super().__init__()
        assert isinstance(data, Mapping)
        # check all elements of dict have same length
        lengths = list(set([len(tmp) for tmp in data.values()]))
        assert len(lengths) == 1
        self.data = data
        self.length = lengths[0]

    def __getitem__(self, index: int) -> dict:
        """Returns the corresponding item for each part of the sample

        Args:
            index: the integer specifying the actual sample

        Returns:
            dict: the sample corresponding to :attr:`index`
        """
        return {k: v[index] for k, v in self.data.items()}

    def __len__(self) -> int:
        """Returns the number of samples in this dataset

        Returns:
            int: the dataset length
        """
        return self.length
