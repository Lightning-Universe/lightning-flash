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
from typing import Any, Callable, Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from flash.data.auto_dataset import AutoDataset
from flash.data.data_pipeline import DataPipeline, Postprocess, Preprocess


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
        train_ds: Optional[AutoDataset] = None,
        valid_ds: Optional[AutoDataset] = None,
        test_ds: Optional[AutoDataset] = None,
        predict_ds: Optional[AutoDataset] = None,
        batch_size: int = 1,
        num_workers: Optional[int] = None,
    ):
        super().__init__()
        self._train_ds = train_ds
        self._valid_ds = valid_ds
        self._test_ds = test_ds
        self._predict_ds = predict_ds

        if self._train_ds is not None:
            self.train_dataloader = self._train_dataloader

        if self._valid_ds is not None:
            self.val_dataloader = self._val_dataloader

        if self._test_ds is not None:
            self.test_dataloader = self._test_dataloader

        if self._predict_ds is not None:
            self.predict_dataloader = self._predict_dataloader

        self.batch_size = batch_size

        # TODO: figure out best solution for setting num_workers
        if num_workers is None:
            if platform.system() == "Darwin":
                num_workers = 0
            else:
                num_workers = os.cpu_count()
        self.num_workers = num_workers

        self._data_pipeline = None
        self._preprocess = None
        self._postprocess = None

        self.setup()

    def setup(self):
        if self._train_ds is not None:
            self._train_ds.setup("train")

        if self._valid_ds is not None:
            self._valid_ds.setup("validation")

        if self._test_ds is not None:
            self._test_ds.setup("test")

        if self._predict_ds is not None:
            self._predict_ds.setup("predict")

    def _train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds if isinstance(self._train_ds, Dataset) else self._train_ds(),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.data_pipeline.worker_preprocessor,
            drop_last=True,
        )

    def _val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._valid_ds if isinstance(self._valid_ds, Dataset) else self._valid_ds(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.data_pipeline.worker_preprocessor,
        )

    def _test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_ds if isinstance(self._test_ds, Dataset) else self._test_ds(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.data_pipeline.worker_preprocessor,
        )

    def _predict_dataloader(self) -> DataLoader:
        predict_ds = self._predict_ds if isinstance(self._predict_ds, Dataset) else self._predict_ds()
        return DataLoader(
            predict_ds,
            batch_size=min(self.batch_size, len(predict_ds)),
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.data_pipeline.worker_preprocessor,
        )

    @property
    def preprocess(self):
        return self._preprocess

    @preprocess.setter
    def preprocess(self, preprocess: Preprocess) -> None:
        self._preprocess = preprocess

    @property
    def postprocess(self):
        return self._postprocess

    @postprocess.setter
    def postprocess(self, postprocess: Postprocess) -> None:
        self._postprocess = postprocess

    @property
    def data_pipeline(self) -> DataPipeline:
        if self._data_pipeline is None:
            preprocess = self._preprocess
            postprocess = self._postprocess
            if preprocess is None and postprocess is None:
                self._data_pipeline = self.default_pipeline()
            return DataPipeline(preprocess, postprocess)
        return self._data_pipeline

    @data_pipeline.setter
    def data_pipeline(self, data_pipeline) -> None:
        self._data_pipeline = data_pipeline
