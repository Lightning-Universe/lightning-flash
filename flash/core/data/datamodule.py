# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import platform
from typing import Any, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from flash.core.data.datapipeline import DataPipeline


class TaskDataPipeline(DataPipeline):

    def after_collate(self, batch: Any) -> Any:
        return (batch["x"], batch["target"]) if isinstance(batch, dict) else batch


class DataModule(pl.LightningDataModule):
    """Basic DataModule class for all Flash tasks

    Args:
        train_ds: Dataset for training. Defaults to None.
        valid_ds: Dataset for validating model performance during training. Defaults to None.
        test_ds: Dataset to test model performance. Defaults to None.
        batch_size: the batch size to be used by the DataLoader. Defaults to 1.
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

        # TODO: figure out best solution for setting num_workers
        if num_workers is None:
            if platform.system() == "Darwin":
                num_workers = 0
            else:
                num_workers = os.cpu_count()
        self.num_workers = num_workers

        self._data_pipeline = None

    def _train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.data_pipeline.collate_fn,
            drop_last=True,
        )

    def _val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._valid_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.data_pipeline.collate_fn,
        )

    def _test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.data_pipeline.collate_fn,
        )

    @property
    def data_pipeline(self) -> DataPipeline:
        if self._data_pipeline is None:
            self._data_pipeline = self.default_pipeline()
        return self._data_pipeline

    @data_pipeline.setter
    def data_pipeline(self, data_pipeline) -> None:
        self._data_pipeline = data_pipeline

    @staticmethod
    def default_pipeline() -> DataPipeline:
        return TaskDataPipeline()
