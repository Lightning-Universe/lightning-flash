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
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, TYPE_CHECKING, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import IterableDataset, Subset
from torch.utils.data.sampler import Sampler

import flash
from flash.core.data.auto_dataset import BaseAutoDataset, IterableAutoDataset
from flash.core.data.base_viz import BaseVisualization
from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_pipeline import DataPipeline
from flash.core.data.io.input import Input
from flash.core.data.io.input_base import InputBase, IterableInput
from flash.core.data.io.input_transform import DefaultInputTransform, InputTransform
from flash.core.data.io.output_transform import OutputTransform
from flash.core.data.splits import SplitDataset
from flash.core.data.utils import _STAGES_PREFIX
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE
from flash.core.utilities.stages import RunningStage

if _FIFTYONE_AVAILABLE and TYPE_CHECKING:
    pass
else:
    SampleCollection = None


class DataModule(pl.LightningDataModule):
    """A basic DataModule class for all Flash tasks. This class includes references to a
    :class:`~flash.core.data.io.input.Input`, :class:`~flash.core.data.io.input_transform.InputTransform`,
    :class:`~flash.core.data.io.output_transform.OutputTransform`, and a
    :class:`~flash.core.data.callback.BaseDataFetcher`.

    Args:
        train_dataset: Dataset for training. Defaults to None.
        val_dataset: Dataset for validating model performance during training. Defaults to None.
        test_dataset: Dataset to test model performance. Defaults to None.
        predict_dataset: Dataset for predicting. Defaults to None.
        input: The :class:`~flash.core.data.io.input.Input` that was used to create the datasets.
        input_transform: The :class:`~flash.core.data.io.input_transform.InputTransform` to use when constructing the
            :class:`~flash.core.data.data_pipeline.DataPipeline`. If ``None``, a
            :class:`~flash.core.data.io.input_transform.DefaultInputTransform` will be used.
        output_transform: The :class:`~flash.core.data.io.output_transform.OutputTransform` to use when constructing the
            :class:`~flash.core.data.data_pipeline.DataPipeline`. If ``None``, a plain
            :class:`~flash.core.data.io.output_transform.OutputTransform` will be used.
        data_fetcher: The :class:`~flash.core.data.callback.BaseDataFetcher` to attach to the
            :class:`~flash.core.data.io.input_transform.InputTransform`. If ``None``, the output from
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

    input_transform_cls = DefaultInputTransform
    output_transform_cls = OutputTransform

    def __init__(
        self,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        predict_dataset: Optional[Dataset] = None,
        input: Optional[Input] = None,
        input_transform: Optional[InputTransform] = None,
        output_transform: Optional[OutputTransform] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: int = 0,
        sampler: Optional[Type[Sampler]] = None,
    ) -> None:

        super().__init__()

        if flash._IS_TESTING and torch.cuda.is_available():
            batch_size = 16

        self._train_ds = train_dataset
        self._val_ds = val_dataset
        self._test_ds = test_dataset
        self._predict_ds = predict_dataset

        if self._train_ds and (val_split is not None and not self._val_ds):
            self._train_ds, self._val_ds = self._split_train_val(self._train_ds, val_split)

        self._input: Input = input
        self._input_transform: Optional[InputTransform] = input_transform
        self._output_transform: Optional[OutputTransform] = output_transform
        self._viz: Optional[BaseVisualization] = None
        self._data_fetcher: Optional[BaseDataFetcher] = data_fetcher or self.configure_data_fetcher()

        # TODO: InputTransform can change
        self.data_fetcher.attach_to_input_transform(self.input_transform)

        if self._train_ds:
            self.train_dataloader = self._train_dataloader

        if self._val_ds:
            self.val_dataloader = self._val_dataloader

        if self._test_ds:
            self.test_dataloader = self._test_dataloader

        if self._predict_ds:
            self.predict_dataloader = self._predict_dataloader

        self.batch_size = batch_size

        if num_workers is None:
            num_workers = 0
        self.num_workers = num_workers

        self.sampler = sampler

        self.set_running_stages()

        # Share state between input objects (this will be available in ``load_sample`` but not in ``load_data``)
        data_pipeline = self.data_pipeline
        data_pipeline.initialize()

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
        """This property returns the data fetcher."""
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
        if isinstance(dataset, (BaseAutoDataset, SplitDataset, InputBase)):
            return self.data_pipeline.worker_input_transform_processor(running_stage)

    def _train_dataloader(self) -> DataLoader:
        """Configure the train dataloader of the datamodule."""
        train_ds: Dataset = self._train_ds() if isinstance(self._train_ds, Callable) else self._train_ds
        shuffle: bool = False
        collate_fn = self._resolve_collate_fn(train_ds, RunningStage.TRAINING)
        if isinstance(train_ds, (IterableAutoDataset, IterableInput)):
            drop_last = False
        else:
            drop_last = len(train_ds) > self.batch_size
        pin_memory = True
        persistent_workers = self.num_workers > 0

        if self.sampler is None:
            sampler = None
            shuffle = not isinstance(train_ds, (IterableDataset, IterableAutoDataset, IterableInput))
        else:
            sampler = self.sampler(train_ds)

        if isinstance(getattr(self, "trainer", None), pl.Trainer):
            return self.trainer.lightning_module.process_train_dataset(
                train_ds,
                trainer=self.trainer,
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
            persistent_workers=persistent_workers,
        )

    def _val_dataloader(self) -> DataLoader:
        """Configure the validation dataloader of the datamodule."""
        val_ds: Dataset = self._val_ds() if isinstance(self._val_ds, Callable) else self._val_ds
        collate_fn = self._resolve_collate_fn(val_ds, RunningStage.VALIDATING)
        pin_memory = True
        persistent_workers = self.num_workers > 0

        if isinstance(getattr(self, "trainer", None), pl.Trainer):
            return self.trainer.lightning_module.process_val_dataset(
                val_ds,
                trainer=self.trainer,
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
            persistent_workers=persistent_workers,
        )

    def _test_dataloader(self) -> DataLoader:
        """Configure the test dataloader of the datamodule."""
        test_ds: Dataset = self._test_ds() if isinstance(self._test_ds, Callable) else self._test_ds
        collate_fn = self._resolve_collate_fn(test_ds, RunningStage.TESTING)
        pin_memory = True
        persistent_workers = False

        if isinstance(getattr(self, "trainer", None), pl.Trainer):
            return self.trainer.lightning_module.process_test_dataset(
                test_ds,
                trainer=self.trainer,
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
            persistent_workers=persistent_workers,
        )

    def _predict_dataloader(self) -> DataLoader:
        """Configure the prediction dataloader of the datamodule."""
        predict_ds: Dataset = self._predict_ds() if isinstance(self._predict_ds, Callable) else self._predict_ds

        if isinstance(predict_ds, (IterableAutoDataset, IterableInput)):
            batch_size = self.batch_size
        else:
            batch_size = min(self.batch_size, len(predict_ds) if len(predict_ds) > 0 else 1)

        collate_fn = self._resolve_collate_fn(predict_ds, RunningStage.PREDICTING)
        pin_memory = True
        persistent_workers = False

        if isinstance(getattr(self, "trainer", None), pl.Trainer):
            return self.trainer.lightning_module.process_predict_dataset(
                predict_ds,
                batch_size=batch_size,
                num_workers=self.num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn,
            )

        return DataLoader(
            predict_ds,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=persistent_workers,
        )

    @property
    def num_classes(self) -> Optional[int]:
        """Property that returns the number of classes of the datamodule if a multiclass task."""
        n_cls_train = getattr(self.train_dataset, "num_classes", None)
        n_cls_val = getattr(self.val_dataset, "num_classes", None)
        n_cls_test = getattr(self.test_dataset, "num_classes", None)
        return n_cls_train or n_cls_val or n_cls_test

    @property
    def multi_label(self) -> Optional[bool]:
        """Property that returns ``True`` if this ``DataModule`` contains multi-label data."""
        multi_label_train = getattr(self.train_dataset, "multi_label", None)
        multi_label_val = getattr(self.val_dataset, "multi_label", None)
        multi_label_test = getattr(self.test_dataset, "multi_label", None)
        return multi_label_train or multi_label_val or multi_label_test

    @property
    def inputs(self) -> Optional[Union[Input, List[InputBase]]]:
        """Property that returns the inputs associated with this ``DataModule``."""
        datasets = [self.train_dataset, self.val_dataset, self.test_dataset, self.predict_dataset]
        inputs = [
            dataset
            for dataset in datasets
            if isinstance(dataset, InputBase)
            or (isinstance(dataset, SplitDataset) and isinstance(dataset.dataset, InputBase))
        ]
        if len(inputs) == 0:
            inputs = self._input
        return inputs

    @property
    def input_transform(self) -> InputTransform:
        """Property that returns the input transform class used on input data."""
        return self._input_transform or self.input_transform_cls()

    @property
    def output_transform(self) -> OutputTransform:
        """Property that returns the :class:`~flash.core.data.io.output_transform.OutputTransform` used to
        output_transform the model outputs."""
        return self._output_transform or self.output_transform_cls()

    @property
    def data_pipeline(self) -> DataPipeline:
        """Property that returns the full data pipeline including the data source, input transform and
        postprocessing."""
        return DataPipeline(self.inputs, self.input_transform, self.output_transform)

    @staticmethod
    def _split_train_val(
        train_dataset: Dataset,
        val_split: float,
    ) -> Tuple[Any, Any]:
        """Utility function for splitting the training dataset into a disjoint subset of training samples and
        validation samples.

        Args:
            train_dataset: A instance of a :class:`torch.utils.data.Dataset`.
            val_split: A float between 0 and 1 determining the number fraction of samples that should go into the
                validation split

        Returns:
            A tuple containing the training and validation datasets
        """

        if not isinstance(val_split, float) or (isinstance(val_split, float) and val_split > 1 or val_split < 0):
            raise MisconfigurationException(f"`val_split` should be a float between 0 and 1. Found {val_split}.")

        if isinstance(train_dataset, (IterableAutoDataset, IterableInput)):
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
