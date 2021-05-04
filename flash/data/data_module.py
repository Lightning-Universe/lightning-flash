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
import pathlib
import platform
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import IterableDataset, Subset

from flash.data.auto_dataset import BaseAutoDataset, IterableAutoDataset
from flash.data.base_viz import BaseVisualization
from flash.data.callback import BaseDataFetcher
from flash.data.data_pipeline import DataPipeline, DefaultPreprocess, Postprocess, Preprocess
from flash.data.data_source import DataSource, FilesDataSource, FoldersDataSource, NumpyDataSource, TensorDataSource
from flash.data.splits import SplitDataset
from flash.data.utils import _STAGES_PREFIX


class DataModule(pl.LightningDataModule):
    """Basic DataModule class for all Flash tasks

    Args:
        train_dataset: Dataset for training. Defaults to None.
        val_dataset: Dataset for validating model performance during training. Defaults to None.
        test_dataset: Dataset to test model performance. Defaults to None.
        predict_dataset: Dataset for predicting. Defaults to None.
        num_workers: The number of workers to use for parallelized loading. Defaults to None.
        batch_size: The batch size to be used by the DataLoader. Defaults to 1.
        num_workers: The number of workers to use for parallelized loading.
            Defaults to None which equals the number of available CPU threads,
            or 0 for Darwin platform.
    """

    preprocess_cls = DefaultPreprocess
    postprocess_cls = Postprocess

    def __init__(
        self,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        predict_dataset: Optional[Dataset] = None,
        data_source: Optional[DataSource] = None,
        preprocess: Optional[Preprocess] = None,
        postprocess: Optional[Postprocess] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        val_split: Optional[float] = None,
        batch_size: int = 1,
        num_workers: Optional[int] = 0,
    ) -> None:

        super().__init__()

        self._data_source: DataSource = data_source
        self._preprocess: Optional[Preprocess] = preprocess
        self._postprocess: Optional[Postprocess] = postprocess
        self._viz: Optional[BaseVisualization] = None
        self._data_fetcher: Optional[BaseDataFetcher] = data_fetcher or self.configure_data_fetcher()

        # TODO: Preprocess can change
        self.data_fetcher.attach_to_preprocess(self.preprocess)

        self._train_ds = train_dataset
        self._val_ds = val_dataset
        self._test_ds = test_dataset
        self._predict_ds = predict_dataset

        if self._train_ds is not None and (val_split is not None and self._val_ds is None):
            self._train_ds, self._val_ds = self._split_train_val(self._train_ds, val_split)

        if self._train_ds:
            self.train_dataloader = self._train_dataloader

        if self._val_ds:
            self.val_dataloader = self._val_dataloader

        if self._test_ds:
            self.test_dataloader = self._test_dataloader

        if self._predict_ds:
            self.predict_dataloader = self._predict_dataloader

        self.batch_size = batch_size

        # TODO: figure out best solution for setting num_workers
        if num_workers is None:
            if platform.system() == "Darwin" or platform.system() == "Windows":
                num_workers = 0
            else:
                num_workers = os.cpu_count()
        self.num_workers = num_workers

        self.set_running_stages()

    @property
    def train_dataset(self) -> Optional[Dataset]:
        """This property returns the train dataset"""
        return self._train_ds

    @property
    def val_dataset(self) -> Optional[Dataset]:
        """This property returns the validation dataset"""
        return self._val_ds

    @property
    def test_dataset(self) -> Optional[Dataset]:
        """This property returns the test dataset"""
        return self._test_ds

    @property
    def predict_dataset(self) -> Optional[Dataset]:
        """This property returns the predict dataset"""
        return self._predict_ds

    @property
    def viz(self) -> BaseVisualization:
        return self._viz or DataModule.configure_data_fetcher()

    @viz.setter
    def viz(self, viz: BaseVisualization) -> None:
        self._viz = viz

    @staticmethod
    def configure_data_fetcher(*args, **kwargs) -> BaseDataFetcher:
        """
        This function is used to configure a :class:`~flash.data.callback.BaseDataFetcher`.
        Override with your custom one.
        """
        return BaseDataFetcher()

    @property
    def data_fetcher(self) -> BaseDataFetcher:
        return self._data_fetcher or DataModule.configure_data_fetcher()

    @data_fetcher.setter
    def data_fetcher(self, data_fetcher: BaseDataFetcher) -> None:
        self._data_fetcher = data_fetcher

    def _reset_iterator(self, stage: str) -> Iterable[Any]:
        iter_name = f"_{stage}_iter"
        # num_workers has to be set to 0 to work properly
        num_workers = self.num_workers
        self.num_workers = 0
        dataloader_fn = getattr(self, f"{stage}_dataloader")
        iterator = iter(dataloader_fn())
        self.num_workers = num_workers
        setattr(self, iter_name, iterator)
        return iterator

    def _show_batch(self, stage: str, func_names: Union[str, List[str]], reset: bool = True) -> None:
        """
        This function is used to handle transforms profiling for batch visualization.
        """
        # don't show in CI
        if os.getenv("FLASH_TESTING", "0") == "1":
            return None
        iter_name = f"_{stage}_iter"

        if not hasattr(self, iter_name):
            self._reset_iterator(stage)

        # list of functions to visualise
        if isinstance(func_names, str):
            func_names = [func_names]

        iter_dataloader = getattr(self, iter_name)
        with self.data_fetcher.enable():
            try:
                _ = next(iter_dataloader)
            except StopIteration:
                iter_dataloader = self._reset_iterator(stage)
                _ = next(iter_dataloader)
            data_fetcher: BaseVisualization = self.data_fetcher
            data_fetcher._show(stage, func_names)
            if reset:
                self.viz.batches[stage] = {}

    def show_train_batch(self, hooks_names: Union[str, List[str]] = 'load_sample', reset: bool = True) -> None:
        """This function is used to visualize a batch from the train dataloader."""
        stage_name: str = _STAGES_PREFIX[RunningStage.TRAINING]
        self._show_batch(stage_name, hooks_names, reset=reset)

    def show_val_batch(self, hooks_names: Union[str, List[str]] = 'load_sample', reset: bool = True) -> None:
        """This function is used to visualize a batch from the validation dataloader."""
        stage_name: str = _STAGES_PREFIX[RunningStage.VALIDATING]
        self._show_batch(stage_name, hooks_names, reset=reset)

    def show_test_batch(self, hooks_names: Union[str, List[str]] = 'load_sample', reset: bool = True) -> None:
        """This function is used to visualize a batch from the test dataloader."""
        stage_name: str = _STAGES_PREFIX[RunningStage.TESTING]
        self._show_batch(stage_name, hooks_names, reset=reset)

    def show_predict_batch(self, hooks_names: Union[str, List[str]] = 'load_sample', reset: bool = True) -> None:
        """This function is used to visualize a batch from the predict dataloader."""
        stage_name: str = _STAGES_PREFIX[RunningStage.PREDICTING]
        self._show_batch(stage_name, hooks_names, reset=reset)

    @staticmethod
    def get_dataset_attribute(dataset: torch.utils.data.Dataset, attr_name: str, default: Optional[Any] = None) -> Any:
        if isinstance(dataset, Subset):
            return getattr(dataset.dataset, attr_name, default)

        return getattr(dataset, attr_name, default)

    @staticmethod
    def set_dataset_attribute(dataset: torch.utils.data.Dataset, attr_name: str, value: Any) -> None:
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
        if isinstance(dataset, (Dataset, IterableDataset)):
            setattr(dataset, attr_name, value)

    def set_running_stages(self):
        if self._train_ds:
            self.set_dataset_attribute(self._train_ds, 'running_stage', RunningStage.TRAINING)

        if self._val_ds:
            self.set_dataset_attribute(self._val_ds, 'running_stage', RunningStage.VALIDATING)

        if self._test_ds:
            self.set_dataset_attribute(self._test_ds, 'running_stage', RunningStage.TESTING)

        if self._predict_ds:
            self.set_dataset_attribute(self._predict_ds, 'running_stage', RunningStage.PREDICTING)

    def _resolve_collate_fn(self, dataset: Dataset, running_stage: RunningStage) -> Optional[Callable]:
        if isinstance(dataset, (BaseAutoDataset, SplitDataset)):
            return self.data_pipeline.worker_preprocessor(running_stage)

    def _train_dataloader(self) -> DataLoader:
        train_ds: Dataset = self._train_ds() if isinstance(self._train_ds, Callable) else self._train_ds
        shuffle = not isinstance(train_ds, (IterableDataset, IterableAutoDataset))
        return DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self._resolve_collate_fn(train_ds, RunningStage.TRAINING)
        )

    def _val_dataloader(self) -> DataLoader:
        val_ds: Dataset = self._val_ds() if isinstance(self._val_ds, Callable) else self._val_ds
        return DataLoader(
            val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._resolve_collate_fn(val_ds, RunningStage.VALIDATING)
        )

    def _test_dataloader(self) -> DataLoader:
        test_ds: Dataset = self._test_ds() if isinstance(self._test_ds, Callable) else self._test_ds
        return DataLoader(
            test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._resolve_collate_fn(test_ds, RunningStage.TESTING)
        )

    def _predict_dataloader(self) -> DataLoader:
        predict_ds: Dataset = self._predict_ds() if isinstance(self._predict_ds, Callable) else self._predict_ds
        if isinstance(predict_ds, IterableAutoDataset):
            batch_size = self.batch_size
        else:
            batch_size = min(self.batch_size, len(predict_ds) if len(predict_ds) > 0 else 1)
        return DataLoader(
            predict_ds,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._resolve_collate_fn(predict_ds, RunningStage.PREDICTING)
        )

    @property
    def num_classes(self) -> Optional[int]:
        return (
            getattr(self.train_dataset, "num_classes", None) or getattr(self.val_dataset, "num_classes", None)
            or getattr(self.test_dataset, "num_classes", None)
        )

    @property
    def data_source(self) -> Optional[DataSource]:
        return self._data_source

    @property
    def preprocess(self) -> Preprocess:
        return self._preprocess or self.preprocess_cls()

    @property
    def postprocess(self) -> Postprocess:
        return self._postprocess or self.postprocess_cls()

    @property
    def data_pipeline(self) -> DataPipeline:
        return DataPipeline(self.data_source, self.preprocess, self.postprocess)

    @staticmethod
    def _split_train_val(
        train_dataset: Dataset,
        val_split: float,
    ) -> Tuple[Any, Any]:

        if not isinstance(val_split, float) or (isinstance(val_split, float) and val_split > 1 or val_split < 0):
            raise MisconfigurationException(f"`val_split` should be a float between 0 and 1. Found {val_split}.")

        if isinstance(train_dataset, IterableAutoDataset):
            raise MisconfigurationException(
                "`val_split` should be `None` when the dataset is built with an IterableDataset."
            )

        train_num_samples = len(train_dataset)
        val_num_samples = int(train_num_samples * val_split)
        val_indices = list(np.random.choice(range(train_num_samples), val_num_samples, replace=False))
        train_indices = [i for i in range(train_num_samples) if i not in val_indices]
        return SplitDataset(train_dataset, train_indices), SplitDataset(train_dataset, val_indices)

    @classmethod
    def from_data_source(
        cls,
        data_source: DataSource,
        train_data: Any = None,
        val_data: Any = None,
        test_data: Any = None,
        predict_data: Any = None,
        train_transform: Optional[Union[str, Dict]] = 'default',
        val_transform: Optional[Union[str, Dict]] = 'default',
        test_transform: Optional[Union[str, Dict]] = 'default',
        predict_transform: Optional[Union[str, Dict]] = 'default',
        data_fetcher: BaseDataFetcher = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        **kwargs: Any,
    ) -> 'DataModule':
        preprocess = preprocess or cls.preprocess_cls(
            train_transform,
            val_transform,
            test_transform,
            predict_transform,
            **kwargs,
        )

        train_dataset, val_dataset, test_dataset, predict_dataset = data_source.to_datasets(
            train_data,
            val_data,
            test_data,
            predict_data,
        )

        return cls(
            train_dataset,
            val_dataset,
            test_dataset,
            predict_dataset,
            data_source=data_source,
            preprocess=preprocess,
            data_fetcher=data_fetcher,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[Union[str, pathlib.Path]] = None,
        val_folder: Optional[Union[str, pathlib.Path]] = None,
        test_folder: Optional[Union[str, pathlib.Path]] = None,
        predict_folder: Union[str, pathlib.Path] = None,
        train_transform: Optional[Union[str, Dict]] = 'default',
        val_transform: Optional[Union[str, Dict]] = 'default',
        test_transform: Optional[Union[str, Dict]] = 'default',
        predict_transform: Optional[Union[str, Dict]] = 'default',
        data_fetcher: BaseDataFetcher = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        **kwargs: Any,
    ) -> 'DataModule':
        data_source = (preprocess or cls.preprocess_cls).data_source_of_type(FoldersDataSource)()

        return cls.from_data_source(
            data_source,
            train_folder,
            val_folder,
            test_folder,
            predict_folder,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    @classmethod
    def from_files(
        cls,
        train_files: Optional[Sequence[Union[str, pathlib.Path]]] = None,
        train_targets: Optional[Sequence[Any]] = None,
        val_files: Optional[Sequence[Union[str, pathlib.Path]]] = None,
        val_targets: Optional[Sequence[Any]] = None,
        test_files: Optional[Sequence[Union[str, pathlib.Path]]] = None,
        test_targets: Optional[Sequence[Any]] = None,
        predict_files: Optional[Sequence[Union[str, pathlib.Path]]] = None,
        train_transform: Optional[Union[str, Dict]] = 'default',
        val_transform: Optional[Union[str, Dict]] = 'default',
        test_transform: Optional[Union[str, Dict]] = 'default',
        predict_transform: Optional[Union[str, Dict]] = 'default',
        data_fetcher: BaseDataFetcher = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        **kwargs: Any,
    ) -> 'DataModule':
        data_source = (preprocess or cls.preprocess_cls).data_source_of_type(FilesDataSource)()

        return cls.from_data_source(
            data_source,
            (train_files, train_targets),
            (val_files, val_targets),
            (test_files, test_targets),
            predict_files,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    @classmethod
    def from_tensors(
        cls,
        train_data: Optional[Collection[torch.Tensor]] = None,
        train_targets: Optional[Collection[Any]] = None,
        val_data: Optional[Collection[torch.Tensor]] = None,
        val_targets: Optional[Sequence[Any]] = None,
        test_data: Optional[Collection[torch.Tensor]] = None,
        test_targets: Optional[Sequence[Any]] = None,
        predict_data: Optional[Collection[torch.Tensor]] = None,
        train_transform: Optional[Union[str, Dict]] = 'default',
        val_transform: Optional[Union[str, Dict]] = 'default',
        test_transform: Optional[Union[str, Dict]] = 'default',
        predict_transform: Optional[Union[str, Dict]] = 'default',
        data_fetcher: BaseDataFetcher = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        **kwargs: Any,
    ) -> 'DataModule':
        data_source = (preprocess or cls.preprocess_cls).data_source_of_type(TensorDataSource)()

        return cls.from_data_source(
            data_source,
            (train_data, train_targets),
            (val_data, val_targets),
            (test_data, test_targets),
            predict_data,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    @classmethod
    def from_numpy(
        cls,
        train_data: Optional[Collection[np.ndarray]] = None,
        train_targets: Optional[Collection[Any]] = None,
        val_data: Optional[Collection[np.ndarray]] = None,
        val_targets: Optional[Sequence[Any]] = None,
        test_data: Optional[Collection[np.ndarray]] = None,
        test_targets: Optional[Sequence[Any]] = None,
        predict_data: Optional[Collection[np.ndarray]] = None,
        train_transform: Optional[Union[str, Dict]] = 'default',
        val_transform: Optional[Union[str, Dict]] = 'default',
        test_transform: Optional[Union[str, Dict]] = 'default',
        predict_transform: Optional[Union[str, Dict]] = 'default',
        data_fetcher: BaseDataFetcher = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        **kwargs: Any,
    ) -> 'DataModule':
        data_source = (preprocess or cls.preprocess_cls).data_source_of_type(NumpyDataSource)()

        return cls.from_data_source(
            data_source,
            (train_data, train_targets),
            (val_data, val_targets),
            (test_data, test_targets),
            predict_data,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
        )
