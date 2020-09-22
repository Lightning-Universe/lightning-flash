from typing import Optional, Union, Any
import os
import warnings
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    """Basic DataModule class for all flash tasks
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
