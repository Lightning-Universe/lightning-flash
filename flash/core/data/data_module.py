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
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import IterableDataset, Subset
from torch.utils.data.sampler import Sampler

import flash
from flash.core.data.auto_dataset import BaseAutoDataset, IterableAutoDataset
from flash.core.data.base_viz import BaseVisualization
from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_pipeline import DataPipeline, DefaultPreprocess, Postprocess, Preprocess
from flash.core.data.data_source import DataSource, DefaultDataSources
from flash.core.data.splits import SplitDataset
from flash.core.data.utils import _STAGES_PREFIX
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE, requires

if _FIFTYONE_AVAILABLE and TYPE_CHECKING:
    from fiftyone.core.collections import SampleCollection
else:
    SampleCollection = None


class DataModule(pl.LightningDataModule):
    """A basic DataModule class for all Flash tasks. This class includes references to a
    :class:`~flash.core.data.data_source.DataSource`, :class:`~flash.core.data.process.Preprocess`,
    :class:`~flash.core.data.process.Postprocess`, and a :class:`~flash.core.data.callback.BaseDataFetcher`.

    Args:
        train_dataset: Dataset for training. Defaults to None.
        val_dataset: Dataset for validating model performance during training. Defaults to None.
        test_dataset: Dataset to test model performance. Defaults to None.
        predict_dataset: Dataset for predicting. Defaults to None.
        data_source: The :class:`~flash.core.data.data_source.DataSource` that was used to create the datasets.
        preprocess: The :class:`~flash.core.data.process.Preprocess` to use when constructing the
            :class:`~flash.core.data.data_pipeline.DataPipeline`. If ``None``, a
            :class:`~flash.core.data.process.DefaultPreprocess` will be used.
        postprocess: The :class:`~flash.core.data.process.Postprocess` to use when constructing the
            :class:`~flash.core.data.data_pipeline.DataPipeline`. If ``None``, a plain
            :class:`~flash.core.data.process.Postprocess` will be used.
        data_fetcher: The :class:`~flash.core.data.callback.BaseDataFetcher` to attach to the
            :class:`~flash.core.data.process.Preprocess`. If ``None``, the output from
            :meth:`~flash.core.data.data_module.DataModule.configure_data_fetcher` will be used.
        val_split: An optional float which gives the relative amount of the training dataset to use for the validation
            dataset.
        batch_size: The batch size to be used by the DataLoader. Defaults to 1.
        num_workers: The number of workers to use for parallelized loading.
            Defaults to None which equals the number of available CPU threads,
            or 0 for Windows or Darwin platform.
        sampler: A sampler following the :class:`~torch.utils.data.sampler.Sampler` type.
            Will be passed to the DataLoader for the training dataset. Defaults to None.
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
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        sampler: Optional[Type[Sampler]] = None,
    ) -> None:

        super().__init__()

        if flash._IS_TESTING and torch.cuda.is_available():
            batch_size = 16

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
            if platform.system() in ("Darwin", "Windows"):
                num_workers = 0
            else:
                num_workers = os.cpu_count()
        self.num_workers = num_workers
        self.sampler = sampler

        self.set_running_stages()

    @property
    def train_dataset(self) -> Optional[Dataset]:
        """This property returns the train dataset."""
        return self._train_ds

    @property
    def val_dataset(self) -> Optional[Dataset]:
        """This property returns the validation dataset."""
        return self._val_ds

    @property
    def test_dataset(self) -> Optional[Dataset]:
        """This property returns the test dataset."""
        return self._test_ds

    @property
    def predict_dataset(self) -> Optional[Dataset]:
        """This property returns the predict dataset."""
        return self._predict_ds

    @property
    def viz(self) -> BaseVisualization:
        return self._viz or DataModule.configure_data_fetcher()

    @viz.setter
    def viz(self, viz: BaseVisualization) -> None:
        self._viz = viz

    @staticmethod
    def configure_data_fetcher(*args, **kwargs) -> BaseDataFetcher:
        """This function is used to configure a :class:`~flash.core.data.callback.BaseDataFetcher`.

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
        """This function is used to handle transforms profiling for batch visualization."""
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
            if reset:
                self.data_fetcher.batches[stage] = {}
            try:
                _ = next(iter_dataloader)
            except StopIteration:
                iter_dataloader = self._reset_iterator(stage)
                _ = next(iter_dataloader)
            data_fetcher: BaseVisualization = self.data_fetcher
            data_fetcher._show(stage, func_names)
            if reset:
                self.data_fetcher.batches[stage] = {}

    def show_train_batch(self, hooks_names: Union[str, List[str]] = "load_sample", reset: bool = True) -> None:
        """This function is used to visualize a batch from the train dataloader."""
        stage_name: str = _STAGES_PREFIX[RunningStage.TRAINING]
        self._show_batch(stage_name, hooks_names, reset=reset)

    def show_val_batch(self, hooks_names: Union[str, List[str]] = "load_sample", reset: bool = True) -> None:
        """This function is used to visualize a batch from the validation dataloader."""
        stage_name: str = _STAGES_PREFIX[RunningStage.VALIDATING]
        self._show_batch(stage_name, hooks_names, reset=reset)

    def show_test_batch(self, hooks_names: Union[str, List[str]] = "load_sample", reset: bool = True) -> None:
        """This function is used to visualize a batch from the test dataloader."""
        stage_name: str = _STAGES_PREFIX[RunningStage.TESTING]
        self._show_batch(stage_name, hooks_names, reset=reset)

    def show_predict_batch(self, hooks_names: Union[str, List[str]] = "load_sample", reset: bool = True) -> None:
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
            self.set_dataset_attribute(self._train_ds, "running_stage", RunningStage.TRAINING)

        if self._val_ds:
            self.set_dataset_attribute(self._val_ds, "running_stage", RunningStage.VALIDATING)

        if self._test_ds:
            self.set_dataset_attribute(self._test_ds, "running_stage", RunningStage.TESTING)

        if self._predict_ds:
            self.set_dataset_attribute(self._predict_ds, "running_stage", RunningStage.PREDICTING)

    def _resolve_collate_fn(self, dataset: Dataset, running_stage: RunningStage) -> Optional[Callable]:
        if isinstance(dataset, (BaseAutoDataset, SplitDataset)):
            return self.data_pipeline.worker_preprocessor(running_stage)

    def _train_dataloader(self) -> DataLoader:
        train_ds: Dataset = self._train_ds() if isinstance(self._train_ds, Callable) else self._train_ds
        shuffle: bool = False
        collate_fn = self._resolve_collate_fn(train_ds, RunningStage.TRAINING)
        if isinstance(train_ds, IterableAutoDataset):
            drop_last = False
        else:
            drop_last = len(train_ds) > self.batch_size
        pin_memory = True

        if self.sampler is None:
            sampler = None
            shuffle = not isinstance(train_ds, (IterableDataset, IterableAutoDataset))
        else:
            sampler = self.sampler(train_ds)

        if isinstance(getattr(self, "trainer", None), pl.Trainer):
            return self.trainer.lightning_module.process_train_dataset(
                train_ds,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=pin_memory,
                shuffle=shuffle,
                drop_last=drop_last,
                collate_fn=collate_fn,
                sampler=sampler,
            )

        return DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn,
        )

    def _val_dataloader(self) -> DataLoader:
        val_ds: Dataset = self._val_ds() if isinstance(self._val_ds, Callable) else self._val_ds
        collate_fn = self._resolve_collate_fn(val_ds, RunningStage.VALIDATING)
        pin_memory = True

        if isinstance(getattr(self, "trainer", None), pl.Trainer):
            return self.trainer.lightning_module.process_val_dataset(
                val_ds,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn,
            )

        return DataLoader(
            val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

    def _test_dataloader(self) -> DataLoader:
        test_ds: Dataset = self._test_ds() if isinstance(self._test_ds, Callable) else self._test_ds
        collate_fn = self._resolve_collate_fn(test_ds, RunningStage.TESTING)
        pin_memory = True

        if isinstance(getattr(self, "trainer", None), pl.Trainer):
            return self.trainer.lightning_module.process_test_dataset(
                test_ds,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn,
            )

        return DataLoader(
            test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

    def _predict_dataloader(self) -> DataLoader:
        predict_ds: Dataset = self._predict_ds() if isinstance(self._predict_ds, Callable) else self._predict_ds
        if isinstance(predict_ds, IterableAutoDataset):
            batch_size = self.batch_size
        else:
            batch_size = min(self.batch_size, len(predict_ds) if len(predict_ds) > 0 else 1)

        collate_fn = self._resolve_collate_fn(predict_ds, RunningStage.PREDICTING)
        pin_memory = True

        if isinstance(getattr(self, "trainer", None), pl.Trainer):
            return self.trainer.lightning_module.process_predict_dataset(
                predict_ds,
                batch_size=batch_size,
                num_workers=self.num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn,
            )

        return DataLoader(
            predict_ds, batch_size=batch_size, num_workers=self.num_workers, pin_memory=True, collate_fn=collate_fn
        )

    @property
    def num_classes(self) -> Optional[int]:
        n_cls_train = getattr(self.train_dataset, "num_classes", None)
        n_cls_val = getattr(self.val_dataset, "num_classes", None)
        n_cls_test = getattr(self.test_dataset, "num_classes", None)
        return n_cls_train or n_cls_val or n_cls_test

    @property
    def multi_label(self) -> Optional[bool]:
        multi_label_train = getattr(self.train_dataset, "multi_label", None)
        multi_label_val = getattr(self.val_dataset, "multi_label", None)
        multi_label_test = getattr(self.test_dataset, "multi_label", None)
        return multi_label_train or multi_label_val or multi_label_test

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

    def available_data_sources(self) -> Sequence[str]:
        """Get the list of available data source names for use with this
        :class:`~flash.core.data.data_module.DataModule`.

        Returns:
            The list of data source names.
        """
        return self.preprocess.available_data_sources()

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

        val_num_samples = int(len(train_dataset) * val_split)
        indices = list(range(len(train_dataset)))
        np.random.shuffle(indices)
        val_indices = indices[:val_num_samples]
        train_indices = indices[val_num_samples:]
        return (
            SplitDataset(train_dataset, train_indices, use_duplicated_indices=True),
            SplitDataset(train_dataset, val_indices, use_duplicated_indices=True),
        )

    @classmethod
    def from_data_source(
        cls,
        data_source: str,
        train_data: Any = None,
        val_data: Any = None,
        test_data: Any = None,
        predict_data: Any = None,
        train_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        val_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        test_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        sampler: Optional[Type[Sampler]] = None,
        **preprocess_kwargs: Any,
    ) -> "DataModule":
        """Creates a :class:`~flash.core.data.data_module.DataModule` object from the given inputs to
        :meth:`~flash.core.data.data_source.DataSource.load_data` (``train_data``, ``val_data``, ``test_data``,
        ``predict_data``). The data source will be resolved from the instantiated
        :class:`~flash.core.data.process.Preprocess`
        using :meth:`~flash.core.data.process.Preprocess.data_source_of_name`.

        Args:
            data_source: The name of the data source to use for the
                :meth:`~flash.core.data.data_source.DataSource.load_data`.
            train_data: The input to :meth:`~flash.core.data.data_source.DataSource.load_data` to use when creating
                the train dataset.
            val_data: The input to :meth:`~flash.core.data.data_source.DataSource.load_data` to use when creating
                the validation dataset.
            test_data: The input to :meth:`~flash.core.data.data_source.DataSource.load_data` to use when creating
                the test dataset.
            predict_data: The input to :meth:`~flash.core.data.data_source.DataSource.load_data` to use when creating
                the predict dataset.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            data_fetcher: The :class:`~flash.core.data.callback.BaseDataFetcher` to pass to the
                :class:`~flash.core.data.data_module.DataModule`.
            preprocess: The :class:`~flash.core.data.data.Preprocess` to pass to the
                :class:`~flash.core.data.data_module.DataModule`. If ``None``, ``cls.preprocess_cls`` will be
                constructed and used.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            batch_size: The ``batch_size`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            num_workers: The ``num_workers`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            sampler: The ``sampler`` to use for the ``train_dataloader``.
            preprocess_kwargs: Additional keyword arguments to use when constructing the preprocess. Will only be used
                if ``preprocess = None``.

        Returns:
            The constructed data module.

        Examples::

            data_module = DataModule.from_data_source(
                DefaultDataSources.FOLDERS,
                train_data="train_folder",
                train_transform={
                    "to_tensor_transform": torch.as_tensor,
                },
            )
        """

        preprocess = preprocess or cls.preprocess_cls(
            train_transform,
            val_transform,
            test_transform,
            predict_transform,
            **preprocess_kwargs,
        )

        data_source = preprocess.data_source_of_name(data_source)

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
            sampler=sampler,
        )

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[str] = None,
        val_folder: Optional[str] = None,
        test_folder: Optional[str] = None,
        predict_folder: Optional[str] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        sampler: Optional[Type[Sampler]] = None,
        **preprocess_kwargs: Any,
    ) -> "DataModule":
        """Creates a :class:`~flash.core.data.data_module.DataModule` object from the given folders using the
        :class:`~flash.core.data.data_source.DataSource` of name
        :attr:`~flash.core.data.data_source.DefaultDataSources.FOLDERS`
        from the passed or constructed :class:`~flash.core.data.process.Preprocess`.

        Args:
            train_folder: The folder containing the train data.
            val_folder: The folder containing the validation data.
            test_folder: The folder containing the test data.
            predict_folder: The folder containing the predict data.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            data_fetcher: The :class:`~flash.core.data.callback.BaseDataFetcher` to pass to the
                :class:`~flash.core.data.data_module.DataModule`.
            preprocess: The :class:`~flash.core.data.data.Preprocess` to pass to the
                :class:`~flash.core.data.data_module.DataModule`. If ``None``, ``cls.preprocess_cls``
                will be constructed and used.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            batch_size: The ``batch_size`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            num_workers: The ``num_workers`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            sampler: The ``sampler`` to use for the ``train_dataloader``.
            preprocess_kwargs: Additional keyword arguments to use when constructing the preprocess. Will only be used
                if ``preprocess = None``.

        Returns:
            The constructed data module.
        """
        return cls.from_data_source(
            DefaultDataSources.FOLDERS,
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
            sampler=sampler,
            **preprocess_kwargs,
        )

    @classmethod
    def from_files(
        cls,
        train_files: Optional[Sequence[str]] = None,
        train_targets: Optional[Sequence[Any]] = None,
        val_files: Optional[Sequence[str]] = None,
        val_targets: Optional[Sequence[Any]] = None,
        test_files: Optional[Sequence[str]] = None,
        test_targets: Optional[Sequence[Any]] = None,
        predict_files: Optional[Sequence[str]] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        sampler: Optional[Type[Sampler]] = None,
        **preprocess_kwargs: Any,
    ) -> "DataModule":
        """Creates a :class:`~flash.core.data.data_module.DataModule` object from the given sequences of files
        using the :class:`~flash.core.data.data_source.DataSource` of name
        :attr:`~flash.core.data.data_source.DefaultDataSources.FILES` from the passed or constructed
        :class:`~flash.core.data.process.Preprocess`.

        Args:
            train_files: A sequence of files to use as the train inputs.
            train_targets: A sequence of targets (one per train file) to use as the train targets.
            val_files: A sequence of files to use as the validation inputs.
            val_targets: A sequence of targets (one per validation file) to use as the validation targets.
            test_files: A sequence of files to use as the test inputs.
            test_targets: A sequence of targets (one per test file) to use as the test targets.
            predict_files: A sequence of files to use when predicting.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            data_fetcher: The :class:`~flash.core.data.callback.BaseDataFetcher` to pass to the
                :class:`~flash.core.data.data_module.DataModule`.
            preprocess: The :class:`~flash.core.data.data.Preprocess` to pass to the
                :class:`~flash.core.data.data_module.DataModule`. If ``None``, ``cls.preprocess_cls``
                will be constructed and used.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            batch_size: The ``batch_size`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            num_workers: The ``num_workers`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            sampler: The ``sampler`` to use for the ``train_dataloader``.
            preprocess_kwargs: Additional keyword arguments to use when constructing the preprocess. Will only be used
                if ``preprocess = None``.

        Returns:
            The constructed data module.
        """
        return cls.from_data_source(
            DefaultDataSources.FILES,
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
            sampler=sampler,
            **preprocess_kwargs,
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
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        sampler: Optional[Type[Sampler]] = None,
        **preprocess_kwargs: Any,
    ) -> "DataModule":
        """Creates a :class:`~flash.core.data.data_module.DataModule` object from the given tensors using the
        :class:`~flash.core.data.data_source.DataSource`
        of name :attr:`~flash.core.data.data_source.DefaultDataSources.TENSOR`
        from the passed or constructed :class:`~flash.core.data.process.Preprocess`.

        Args:
            train_data: A tensor or collection of tensors to use as the train inputs.
            train_targets: A sequence of targets (one per train input) to use as the train targets.
            val_data: A tensor or collection of tensors to use as the validation inputs.
            val_targets: A sequence of targets (one per validation input) to use as the validation targets.
            test_data: A tensor or collection of tensors to use as the test inputs.
            test_targets: A sequence of targets (one per test input) to use as the test targets.
            predict_data: A tensor or collection of tensors to use when predicting.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            data_fetcher: The :class:`~flash.core.data.callback.BaseDataFetcher` to pass to the
                :class:`~flash.core.data.data_module.DataModule`.
            preprocess: The :class:`~flash.core.data.data.Preprocess` to pass to the
                :class:`~flash.core.data.data_module.DataModule`. If ``None``, ``cls.preprocess_cls``
                will be constructed and used.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            batch_size: The ``batch_size`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            num_workers: The ``num_workers`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            sampler: The ``sampler`` to use for the ``train_dataloader``.
            preprocess_kwargs: Additional keyword arguments to use when constructing the preprocess. Will only be used
                if ``preprocess = None``.

        Returns:
            The constructed data module.

        Examples::

            data_module = DataModule.from_tensors(
                train_files=torch.rand(3, 128),
                train_targets=[1, 0, 1],
                train_transform={
                    "to_tensor_transform": torch.as_tensor,
                },
            )
        """
        return cls.from_data_source(
            DefaultDataSources.TENSORS,
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
            sampler=sampler,
            **preprocess_kwargs,
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
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        sampler: Optional[Type[Sampler]] = None,
        **preprocess_kwargs: Any,
    ) -> "DataModule":
        """Creates a :class:`~flash.core.data.data_module.DataModule` object from the given numpy array using the
        :class:`~flash.core.data.data_source.DataSource`
        of name :attr:`~flash.core.data.data_source.DefaultDataSources.NUMPY`
        from the passed or constructed :class:`~flash.core.data.process.Preprocess`.

        Args:
            train_data: A numpy array to use as the train inputs.
            train_targets: A sequence of targets (one per train input) to use as the train targets.
            val_data: A numpy array to use as the validation inputs.
            val_targets: A sequence of targets (one per validation input) to use as the validation targets.
            test_data: A numpy array to use as the test inputs.
            test_targets: A sequence of targets (one per test input) to use as the test targets.
            predict_data: A numpy array to use when predicting.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            data_fetcher: The :class:`~flash.core.data.callback.BaseDataFetcher` to pass to the
                :class:`~flash.core.data.data_module.DataModule`.
            preprocess: The :class:`~flash.core.data.data.Preprocess` to pass to the
                :class:`~flash.core.data.data_module.DataModule`. If ``None``, ``cls.preprocess_cls``
                will be constructed and used.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            batch_size: The ``batch_size`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            num_workers: The ``num_workers`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            sampler: The ``sampler`` to use for the ``train_dataloader``.
            preprocess_kwargs: Additional keyword arguments to use when constructing the preprocess. Will only be used
                if ``preprocess = None``.

        Returns:
            The constructed data module.

        Examples::

            data_module = DataModule.from_numpy(
                train_files=np.random.rand(3, 128),
                train_targets=[1, 0, 1],
                train_transform={
                    "to_tensor_transform": torch.as_tensor,
                },
            )
        """
        return cls.from_data_source(
            DefaultDataSources.NUMPY,
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
            sampler=sampler,
            **preprocess_kwargs,
        )

    @classmethod
    def from_json(
        cls,
        input_fields: Union[str, Sequence[str]],
        target_fields: Optional[Union[str, Sequence[str]]] = None,
        train_file: Optional[str] = None,
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        predict_file: Optional[str] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        sampler: Optional[Type[Sampler]] = None,
        field: Optional[str] = None,
        **preprocess_kwargs: Any,
    ) -> "DataModule":
        """Creates a :class:`~flash.core.data.data_module.DataModule` object from the given JSON files using the
        :class:`~flash.core.data.data_source.DataSource`
        of name :attr:`~flash.core.data.data_source.DefaultDataSources.JSON`
        from the passed or constructed :class:`~flash.core.data.process.Preprocess`.

        Args:
            input_fields: The field or fields in the JSON objects to use for the input.
            target_fields: The field or fields in the JSON objects to use for the target.
            train_file: The JSON file containing the training data.
            val_file: The JSON file containing the validation data.
            test_file: The JSON file containing the testing data.
            predict_file: The JSON file containing the data to use when predicting.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            data_fetcher: The :class:`~flash.core.data.callback.BaseDataFetcher` to pass to the
                :class:`~flash.core.data.data_module.DataModule`.
            preprocess: The :class:`~flash.core.data.data.Preprocess` to pass to the
                :class:`~flash.core.data.data_module.DataModule`. If ``None``, ``cls.preprocess_cls``
                will be constructed and used.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            batch_size: The ``batch_size`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            num_workers: The ``num_workers`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            sampler: The ``sampler`` to use for the ``train_dataloader``.
            field: To specify the field that holds the data in the JSON file.
            preprocess_kwargs: Additional keyword arguments to use when constructing the preprocess. Will only be used
                if ``preprocess = None``.

        Returns:
            The constructed data module.

        Examples::

            data_module = DataModule.from_json(
                "input",
                "target",
                train_file="train_data.json",
                train_transform={
                    "to_tensor_transform": torch.as_tensor,
                },
            )

            # In the case where the data is of the form:
            # {
            #     "version": 0.0.x,
            #     "data": [
            #         {
            #             "input_field" : "input_data",
            #             "target_field" : "target_output"
            #         },
            #         ...
            #     ]
            # }

            data_module = DataModule.from_json(
                "input",
                "target",
                train_file="train_data.json",
                train_transform={
                    "to_tensor_transform": torch.as_tensor,
                },
                feild="data"
            )
        """
        return cls.from_data_source(
            DefaultDataSources.JSON,
            (train_file, input_fields, target_fields, field),
            (val_file, input_fields, target_fields, field),
            (test_file, input_fields, target_fields, field),
            (predict_file, input_fields, target_fields, field),
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            **preprocess_kwargs,
        )

    @classmethod
    def from_csv(
        cls,
        input_fields: Union[str, Sequence[str]],
        target_fields: Optional[Union[str, Sequence[str]]] = None,
        train_file: Optional[str] = None,
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        predict_file: Optional[str] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        sampler: Optional[Type[Sampler]] = None,
        **preprocess_kwargs: Any,
    ) -> "DataModule":
        """Creates a :class:`~flash.core.data.data_module.DataModule` object from the given CSV files using the
        :class:`~flash.core.data.data_source.DataSource`
        of name :attr:`~flash.core.data.data_source.DefaultDataSources.CSV`
        from the passed or constructed :class:`~flash.core.data.process.Preprocess`.

        Args:
            input_fields: The field or fields (columns) in the CSV file to use for the input.
            target_fields: The field or fields (columns) in the CSV file to use for the target.
            train_file: The CSV file containing the training data.
            val_file: The CSV file containing the validation data.
            test_file: The CSV file containing the testing data.
            predict_file: The CSV file containing the data to use when predicting.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            data_fetcher: The :class:`~flash.core.data.callback.BaseDataFetcher` to pass to the
                :class:`~flash.core.data.data_module.DataModule`.
            preprocess: The :class:`~flash.core.data.data.Preprocess` to pass to the
                :class:`~flash.core.data.data_module.DataModule`. If ``None``, ``cls.preprocess_cls``
                will be constructed and used.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            batch_size: The ``batch_size`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            num_workers: The ``num_workers`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            sampler: The ``sampler`` to use for the ``train_dataloader``.
            preprocess_kwargs: Additional keyword arguments to use when constructing the preprocess. Will only be used
                if ``preprocess = None``.

        Returns:
            The constructed data module.

        Examples::

            data_module = DataModule.from_csv(
                "input",
                "target",
                train_file="train_data.csv",
                train_transform={
                    "to_tensor_transform": torch.as_tensor,
                },
            )
        """
        return cls.from_data_source(
            DefaultDataSources.CSV,
            (train_file, input_fields, target_fields),
            (val_file, input_fields, target_fields),
            (test_file, input_fields, target_fields),
            (predict_file, input_fields, target_fields),
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            **preprocess_kwargs,
        )

    @classmethod
    def from_datasets(
        cls,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        predict_dataset: Optional[Dataset] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        sampler: Optional[Type[Sampler]] = None,
        **preprocess_kwargs: Any,
    ) -> "DataModule":
        """Creates a :class:`~flash.core.data.data_module.DataModule` object from the given datasets using the
        :class:`~flash.core.data.data_source.DataSource`
        of name :attr:`~flash.core.data.data_source.DefaultDataSources.DATASETS`
        from the passed or constructed :class:`~flash.core.data.process.Preprocess`.

        Args:
            train_dataset: Dataset used during training.
            val_dataset: Dataset used during validating.
            test_dataset: Dataset used during testing.
            predict_dataset: Dataset used during predicting.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            data_fetcher: The :class:`~flash.core.data.callback.BaseDataFetcher` to pass to the
                :class:`~flash.core.data.data_module.DataModule`.
            preprocess: The :class:`~flash.core.data.data.Preprocess` to pass to the
                :class:`~flash.core.data.data_module.DataModule`. If ``None``, ``cls.preprocess_cls``
                will be constructed and used.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            batch_size: The ``batch_size`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            num_workers: The ``num_workers`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            sampler: The ``sampler`` to use for the ``train_dataloader``.
            preprocess_kwargs: Additional keyword arguments to use when constructing the preprocess. Will only be used
                if ``preprocess = None``.

        Returns:
            The constructed data module.

        Examples::

            data_module = DataModule.from_datasets(
                train_dataset=train_dataset,
                train_transform={
                    "to_tensor_transform": torch.as_tensor,
                },
            )
        """
        return cls.from_data_source(
            DefaultDataSources.DATASETS,
            train_dataset,
            val_dataset,
            test_dataset,
            predict_dataset,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            **preprocess_kwargs,
        )

    @classmethod
    @requires("fiftyone")
    def from_fiftyone(
        cls,
        train_dataset: Optional[SampleCollection] = None,
        val_dataset: Optional[SampleCollection] = None,
        test_dataset: Optional[SampleCollection] = None,
        predict_dataset: Optional[SampleCollection] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        **preprocess_kwargs: Any,
    ) -> "DataModule":
        """Creates a :class:`~flash.core.data.data_module.DataModule` object
        from the given FiftyOne Datasets using the
        :class:`~flash.core.data.data_source.DataSource` of name
        :attr:`~flash.core.data.data_source.DefaultDataSources.FIFTYONE`
        from the passed or constructed :class:`~flash.core.data.process.Preprocess`.

        Args:
            train_dataset: The ``fiftyone.core.collections.SampleCollection`` containing the train data.
            val_dataset: The ``fiftyone.core.collections.SampleCollection`` containing the validation data.
            test_dataset: The ``fiftyone.core.collections.SampleCollection`` containing the test data.
            predict_dataset: The ``fiftyone.core.collections.SampleCollection`` containing the predict data.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            data_fetcher: The :class:`~flash.core.data.callback.BaseDataFetcher` to pass to the
                :class:`~flash.core.data.data_module.DataModule`.
            preprocess: The :class:`~flash.core.data.data.Preprocess` to pass to the
                :class:`~flash.core.data.data_module.DataModule`. If ``None``, ``cls.preprocess_cls``
                will be constructed and used.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            batch_size: The ``batch_size`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            num_workers: The ``num_workers`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            preprocess_kwargs: Additional keyword arguments to use when constructing the preprocess. Will only be used
                if ``preprocess = None``.

        Returns:
            The constructed data module.

        Examples::

            train_dataset = fo.Dataset.from_dir(
                "/path/to/dataset",
                dataset_type=fo.types.ImageClassificationDirectoryTree,
            )
            data_module = DataModule.from_fiftyone(
                train_data = train_dataset,
                train_transform={
                    "to_tensor_transform": torch.as_tensor,
                },
            )
        """
        return cls.from_data_source(
            DefaultDataSources.FIFTYONE,
            train_dataset,
            val_dataset,
            test_dataset,
            predict_dataset,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            **preprocess_kwargs,
        )
