from typing import Optional, Union, Any
import pathlib
import pickle
import os
import warnings

from pytorch_lightning.core.datamodule import LightningDataModule

import torch
from torch.utils.data import DataLoader, Dataset


class BaseData(object):
    """Basic Data object providing helper functions to load various file formats

    Raises:
        RuntimeError: Trying to load a not supported file format
        ImportError: numpy is not available
        ValueError: Invalid file format for numpy

    Returns:
        [type]: [description]
    """

    @staticmethod
    def load_file(path: Union[str, pathlib.Path], **kwargs) -> torch.Tensor:
        """loads a file and maps the correct way to load based on the file extension

        Args:
            path: the file to load

        Raises:
            RuntimeError: not supported file format

        Returns:
            torch.Tensor: tensor containing the loaded file content
        """
        path = str(path)
        if path.endswith(".pt"):
            loaded = BaseData.load_torch(path, **kwargs)

        elif any([path.endswith(ext) for ext in [".npy", ".npz", ".txt"]]):
            loaded = BaseData.load_numpy(path, **kwargs)

        elif path.endswith(".pkl"):
            loaded = BaseData.load_pickle(path, **kwargs)

        else:
            raise RuntimeError

        return torch.utils.data._utils.collate.default_convert(loaded)

    @staticmethod
    def load_torch(path, **kwargs) -> Union[Any, torch.Tensor]:
        """Loads objects that have been saved with ``torch.save``

        Args:
            path: the file to load

        Returns:
            torch.Tensor: the loaded file contents
        """
        return torch.load(path, **kwargs)

    @staticmethod
    def load_numpy(path, **kwargs) -> torch.Tensor:
        """loading objects with numpy

        Args:
            path: the file to load

        Raises:
            ImportError: numpy is not available
            ValueError: invalid file extension

        Returns:
            torch.Tensor: the loaded file contents
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy is not available. Please install it with `pip install numpy`")

        if path.endswith(".npy") or path.endswith(".npz"):
            loaded = np.load(path, **kwargs)

        elif path.endswith(".txt"):
            loaded = np.loadtxt(path, **kwargs)

        else:
            raise ValueError

        return torch.from_numpy(loaded)

    @staticmethod
    def load_pickle(path, **kwargs) -> Union[Any, torch.Tensor]:
        """loads objects that have been saved with pickle

        Args:
            path: the file to load

        Returns:
            Union[Any, torch.Tensor]: the loaded file contents
        """

        with open(path, "rb") as f:
            return pickle.load(f, **kwargs)


class FlashDataModule(LightningDataModule):
    """Basic DataModule for all flash tasks

    Args:
        train_ds: Dataset for training. Defaults to None.
        valid_ds: Dataset for validating model performance during training. Defaults to None.
        test_ds: Dataset to test model performance. Defaults to None.
        batch_size: the batchsize to be used by the dataloader. Defaults to 1.
        num_workers: The number of workers to use for parallelized loading.
            Defaults to None which equals the number of available CPU threads.
    """

    def __init__(
        self,
        train_ds: Optional[Dataset] = None,
        valid_ds: Optional[Dataset] = None,
        test_ds: Optional[Dataset] = None,
        batch_size: int = 1,
        num_workers: Optional[int] = None,
    ):

        super().__init__()
        self._train_ds = train_ds
        self._valid_ds = valid_ds
        self._test_ds = test_ds

        if self._train_ds is not None:
            self.train_dataloader = self._train_dataloader

        if self._valid_ds is not None:
            self.val_dataloader = self._val_dataloader

        if self._test_ds is not None:
            self.test_dataloader = self._test_dataloader

        self.batch_size = batch_size

        if num_workers is None:
            num_workers = os.cpu_count()

        if num_workers is None:
            warnings.warn("could not infer cpu count automatically, setting it to zero")
            num_workers = 0
        self.num_workers = num_workers

    def _train_dataloader(self):
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def _val_dataloader(self):
        return DataLoader(
            self._valid_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def _test_dataloader(self):
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
