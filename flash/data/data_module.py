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
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from datasets.splits import SplitInfo
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import IterableDataset, Subset

from flash.data.auto_dataset import AutoDataset, BaseAutoDataset, IterableAutoDataset
from flash.data.base_viz import BaseVisualization
from flash.data.callback import BaseDataFetcher
from flash.data.data_pipeline import DataPipeline, Postprocess, Preprocess
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

    preprocess_cls = Preprocess
    postprocess_cls = Postprocess

    def __init__(
        self,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        predict_dataset: Optional[Dataset] = None,
        batch_size: int = 1,
        num_workers: Optional[int] = 0,
    ) -> None:

        super().__init__()
        self._train_ds = train_dataset
        self._val_ds = val_dataset
        self._test_ds = test_dataset
        self._predict_ds = predict_dataset

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

        self._preprocess: Optional[Preprocess] = None
        self._postprocess: Optional[Postprocess] = None
        self._viz: Optional[BaseVisualization] = None
        self._data_fetcher: Optional[BaseDataFetcher] = None

        # this may also trigger data preloading
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

    def _reset_iterator(self, stage: RunningStage) -> Iterable[Any]:
        iter_name = f"_{stage}_iter"
        # num_workers has to be set to 0 to work properly
        num_workers = self.num_workers
        self.num_workers = 0
        dataloader_fn = getattr(self, f"{stage}_dataloader")
        iterator = iter(dataloader_fn())
        self.num_workers = num_workers
        setattr(self, iter_name, iterator)
        return iterator

    def _show_batch(self, stage: RunningStage, func_names: Union[str, List[str]], reset: bool = True) -> None:
        """
        This function is used to handle transforms profiling for batch visualization.
        """
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

    def generate_auto_dataset(self, *args, **kwargs):
        if all(a is None for a in args) and len(kwargs) == 0:
            return None
        return self.data_pipeline._generate_auto_dataset(*args, **kwargs)

    @property
    def num_classes(self) -> Optional[int]:
        return (
            getattr(self.train_dataset, "num_classes", None) or getattr(self.val_dataset, "num_classes", None)
            or getattr(self.test_dataset, "num_classes", None)
        )

    @property
    def preprocess(self) -> Preprocess:
        return self._preprocess or self.preprocess_cls()

    @property
    def postprocess(self) -> Postprocess:
        return self._postprocess or self.postprocess_cls()

    @property
    def data_pipeline(self) -> DataPipeline:
        return DataPipeline(self.preprocess, self.postprocess)

    @staticmethod
    def _check_transforms(transform: Dict[str, Union[Module, Callable]]) -> Dict[str, Union[Module, Callable]]:
        if not isinstance(transform, dict):
            raise MisconfigurationException(
                "Transform should be a dict. Here are the available keys "
                f"for your transforms: {DataPipeline.PREPROCESS_FUNCS}."
            )
        return transform

    @classmethod
    def autogenerate_dataset(
        cls,
        data: Any,
        running_stage: RunningStage,
        whole_data_load_fn: Optional[Callable] = None,
        per_sample_load_fn: Optional[Callable] = None,
        data_pipeline: Optional[DataPipeline] = None,
        use_iterable_auto_dataset: bool = False,
    ) -> BaseAutoDataset:
        """
        This function is used to generate an ``BaseAutoDataset`` from a ``DataPipeline`` if provided
        or from the provided ``whole_data_load_fn``, ``per_sample_load_fn`` functions directly
        """

        preprocess = getattr(data_pipeline, '_preprocess_pipeline', None)

        if whole_data_load_fn is None:
            whole_data_load_fn = getattr(
                preprocess,
                DataPipeline._resolve_function_hierarchy('load_data', preprocess, running_stage, Preprocess)
            )

        if per_sample_load_fn is None:
            per_sample_load_fn = getattr(
                preprocess,
                DataPipeline._resolve_function_hierarchy('load_sample', preprocess, running_stage, Preprocess)
            )
        if use_iterable_auto_dataset:
            return IterableAutoDataset(
                data, whole_data_load_fn, per_sample_load_fn, data_pipeline, running_stage=running_stage
            )
        return BaseAutoDataset(data, whole_data_load_fn, per_sample_load_fn, data_pipeline, running_stage=running_stage)

    @classmethod
    def _split_train_val(
        cls,
        train_dataset: Union[AutoDataset, IterableAutoDataset],
        val_split: float,
    ) -> Tuple[Any, Any]:

        if not isinstance(val_split, float) or (isinstance(val_split, float) and val_split > 1 or val_split < 0):
            raise MisconfigurationException("`val_split` should be a float between 0 and 1.")

        if isinstance(train_dataset, IterableAutoDataset):
            raise MisconfigurationException(
                "`val_split` should be `None` when the dataset is built with an IterativeDataset."
            )

        train_num_samples = len(train_dataset)
        val_num_samples = int(train_num_samples * val_split)
        val_indices = list(np.random.choice(range(train_num_samples), val_num_samples, replace=False))
        train_indices = [i for i in range(train_num_samples) if i not in val_indices]
        return SplitDataset(train_dataset, train_indices), SplitDataset(train_dataset, val_indices)

    @classmethod
    def _generate_dataset_if_possible(
        cls,
        data: Optional[Any],
        running_stage: RunningStage,
        whole_data_load_fn: Optional[Callable] = None,
        per_sample_load_fn: Optional[Callable] = None,
        data_pipeline: Optional[DataPipeline] = None,
        use_iterable_auto_dataset: bool = False,
    ) -> Optional[BaseAutoDataset]:
        if data is None:
            return

        if data_pipeline:
            return data_pipeline._generate_auto_dataset(
                data,
                running_stage=running_stage,
                use_iterable_auto_dataset=use_iterable_auto_dataset,
            )

        return cls.autogenerate_dataset(
            data,
            running_stage,
            whole_data_load_fn,
            per_sample_load_fn,
            data_pipeline,
            use_iterable_auto_dataset=use_iterable_auto_dataset,
        )

    @classmethod
    def from_load_data_inputs(
        cls,
        train_load_data_input: Optional[Any] = None,
        val_load_data_input: Optional[Any] = None,
        test_load_data_input: Optional[Any] = None,
        predict_load_data_input: Optional[Any] = None,
        data_fetcher: BaseDataFetcher = None,
        preprocess: Optional[Preprocess] = None,
        postprocess: Optional[Postprocess] = None,
        use_iterable_auto_dataset: bool = False,
        seed: int = 42,
        val_split: float = 0,
        **kwargs,
    ) -> 'DataModule':
        """
        This functions is an helper to generate a ``DataModule`` from a ``DataPipeline``.

        Args:
            cls: ``DataModule`` subclass
            train_load_data_input: Data to be received by the ``train_load_data`` function
                from this :class:`~flash.data.process.Preprocess`
            val_load_data_input: Data to be received by the ``val_load_data`` function
                from this :class:`~flash.data.process.Preprocess`
            test_load_data_input: Data to be received by the ``test_load_data`` function
                from this :class:`~flash.data.process.Preprocess`
            predict_load_data_input: Data to be received by the ``predict_load_data`` function
                from this :class:`~flash.data.process.Preprocess`
            kwargs: Any extra arguments to instantiate the provided ``DataModule``
        """
        # trick to get data_pipeline from empty DataModule
        if preprocess or postprocess:
            data_pipeline = DataPipeline(
                preprocess or cls(**kwargs).preprocess,
                postprocess or cls(**kwargs).postprocess,
            )
        else:
            data_pipeline = cls(**kwargs).data_pipeline

        data_fetcher: BaseDataFetcher = data_fetcher or cls.configure_data_fetcher()

        data_fetcher.attach_to_preprocess(data_pipeline._preprocess_pipeline)

        train_dataset = cls._generate_dataset_if_possible(
            train_load_data_input,
            running_stage=RunningStage.TRAINING,
            data_pipeline=data_pipeline,
            use_iterable_auto_dataset=use_iterable_auto_dataset,
        )
        val_dataset = cls._generate_dataset_if_possible(
            val_load_data_input,
            running_stage=RunningStage.VALIDATING,
            data_pipeline=data_pipeline,
            use_iterable_auto_dataset=use_iterable_auto_dataset,
        )
        test_dataset = cls._generate_dataset_if_possible(
            test_load_data_input,
            running_stage=RunningStage.TESTING,
            data_pipeline=data_pipeline,
            use_iterable_auto_dataset=use_iterable_auto_dataset,
        )
        predict_dataset = cls._generate_dataset_if_possible(
            predict_load_data_input,
            running_stage=RunningStage.PREDICTING,
            data_pipeline=data_pipeline,
            use_iterable_auto_dataset=use_iterable_auto_dataset,
        )

        if train_dataset is not None and (val_split is not None and val_dataset is None):
            train_dataset, val_dataset = cls._split_train_val(train_dataset, val_split)

        datamodule = cls(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            predict_dataset=predict_dataset,
            **kwargs
        )
        datamodule._preprocess = data_pipeline._preprocess_pipeline
        datamodule._postprocess = data_pipeline._postprocess_pipeline
        data_fetcher.attach_to_datamodule(datamodule)
        return datamodule
