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
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataset import IterableDataset
from torch.utils.data.sampler import Sampler

import flash
from flash.core.data.base_viz import BaseVisualization
from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_pipeline import DataPipeline, DataPipelineState
from flash.core.data.io.input import DataKeys, Input, InputBase, IterableInput
from flash.core.data.io.input_transform import _InputTransformProcessorV2, InputTransform
from flash.core.data.io.output_transform import OutputTransform
from flash.core.data.splits import SplitDataset
from flash.core.data.utils import _STAGES_PREFIX
from flash.core.registry import FlashRegistry
from flash.core.utilities.stages import RunningStage


class DatasetInput(Input):
    """The ``DatasetInput`` implements default behaviours for data sources which expect the input to
    :meth:`~flash.core.data.io.input.Input.load_data` to be a :class:`torch.utils.data.dataset.Dataset`
    """

    def load_sample(self, sample: Any) -> Dict[str, Any]:
        if isinstance(sample, tuple) and len(sample) == 2:
            return {DataKeys.INPUT: sample[0], DataKeys.TARGET: sample[1]}
        return {DataKeys.INPUT: sample}


class DataModule(pl.LightningDataModule):
    """A basic DataModule class for all Flash tasks. This class includes references to a
    :class:`~flash.core.data.datasets.Input` and a :class:`~flash.core.data.callback.BaseDataFetcher`.

    Args:
        train_input: Input dataset for training. Defaults to None.
        val_input: Input dataset for validating model performance during training. Defaults to None.
        test_input: Input dataset to test model performance. Defaults to None.
        predict_input: Input dataset for predicting. Defaults to None.
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

    input_transform_cls = InputTransform
    output_transform_cls = OutputTransform
    input_transforms_registry: Optional[FlashRegistry] = None

    def __init__(
        self,
        train_input: Optional[Input] = None,
        val_input: Optional[Input] = None,
        test_input: Optional[Input] = None,
        predict_input: Optional[Input] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        val_split: Optional[float] = None,
        batch_size: Optional[int] = None,
        num_workers: int = 0,
        sampler: Optional[Type[Sampler]] = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        output_transform: Optional[OutputTransform] = None,
    ) -> None:

        if not batch_size:
            raise MisconfigurationException("The `batch_size` should be provided to the DataModule on instantiation.")

        if flash._IS_TESTING and torch.cuda.is_available():
            batch_size = 16

        self._input_transform: Optional[OutputTransform] = None
        self._output_transform: Optional[OutputTransform] = output_transform
        self._viz: Optional[BaseVisualization] = None

        self._train_input = train_input
        self._val_input = val_input
        self._test_input = test_input
        self._predict_input = predict_input

        if self._train_input and self._val_input and isinstance(val_split, float) and val_split > 0:
            raise MisconfigurationException(
                "A `val_dataset` was provided with `val_split`. Please, choose one or the other."
            )

        if self._train_input and (val_split is not None and not self._val_input):
            self._train_input, self._val_input = self._split_train_val(self._train_input, val_split)

        self._data_fetcher: Optional[BaseDataFetcher] = data_fetcher or self.configure_data_fetcher()

        self._train_dataloader_collate_fn = self._resolve_dataloader_collate_fn(self._train_input)
        self._val_dataloader_collate_fn = self._resolve_dataloader_collate_fn(self._val_input)
        self._test_dataloader_collate_fn = self._resolve_dataloader_collate_fn(self._test_input)
        self._predict_dataloader_collate_fn = self._resolve_dataloader_collate_fn(self._predict_input)

        self._train_on_after_batch_transfer_fn = self._resolve_on_after_batch_transfer_fn(self._train_input)
        self._val_on_after_batch_transfer_fn = self._resolve_on_after_batch_transfer_fn(self._val_input)
        self._test_on_after_batch_transfer_fn = self._resolve_on_after_batch_transfer_fn(self._test_input)
        self._predict_on_after_batch_transfer_fn = self._resolve_on_after_batch_transfer_fn(self._predict_input)

        if self._train_input:
            self.train_dataloader = self._train_dataloader

        if self._val_input:
            self.val_dataloader = self._val_dataloader

        if self._test_input:
            self.test_dataloader = self._test_dataloader

        if self._predict_input:
            self.predict_dataloader = self._predict_dataloader

        self.batch_size = batch_size

        if num_workers is None:
            num_workers = 0
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers and num_workers > 0
        self.pin_memory = pin_memory

        self.sampler = sampler

        super().__init__(self)

    @property
    def train_dataset(self) -> Optional[Input]:
        """This property returns the train dataset."""
        return self._train_input

    @property
    def val_dataset(self) -> Optional[Input]:
        """This property returns the validation dataset."""
        return self._val_input

    @property
    def test_dataset(self) -> Optional[Input]:
        """This property returns the test dataset."""
        return self._test_input

    @property
    def predict_dataset(self) -> Optional[Input]:
        """This property returns the predict dataset."""
        return self._predict_input

    def _resolve_transform(self, ds: Optional[Input]) -> Optional[InputTransform]:
        if not isinstance(ds, Input):
            return None
        return ds.transform

    def _resolve_dataloader_collate_fn(self, ds: Optional[Input]) -> Optional[Callable]:
        if not ds:
            return None
        if isinstance(ds.transform, InputTransform):
            return ds._create_dataloader_collate_fn([self.data_fetcher])
        return default_collate

    def _resolve_on_after_batch_transfer_fn(self, ds: Optional[Input]) -> Optional[Callable]:
        if not ds:
            return None
        if isinstance(ds.transform, InputTransform):
            return ds._create_on_after_batch_transfer_fn([self.data_fetcher])

    def _train_dataloader(self) -> DataLoader:
        if isinstance(getattr(self, "trainer", None), pl.Trainer):
            if isinstance(self.trainer.lightning_module, flash.Task):
                self.connect(self.trainer.lightning_module)

        train_ds: Input = self._train_input
        collate_fn = self._train_dataloader_collate_fn

        transform_processor = None
        if isinstance(collate_fn, _InputTransformProcessorV2):
            transform_processor = collate_fn
            collate_fn = transform_processor.collate_fn

        shuffle: bool = False
        if isinstance(train_ds, IterableDataset):
            drop_last = False
        else:
            drop_last = len(train_ds) > self.batch_size

        if self.sampler is None:
            sampler = None
            shuffle = not isinstance(train_ds, IterableDataset)
        else:
            sampler = self.sampler(train_ds)

        if isinstance(getattr(self, "trainer", None), pl.Trainer):
            dataloader = self.trainer.lightning_module.process_train_dataset(
                train_ds,
                trainer=self.trainer,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=shuffle,
                drop_last=drop_last,
                collate_fn=collate_fn,
                sampler=sampler,
            )
        else:
            dataloader = DataLoader(
                train_ds,
                batch_size=self.batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=drop_last,
                collate_fn=collate_fn,
                persistent_workers=self.persistent_workers,
            )

        if transform_processor is not None:
            transform_processor.collate_fn = dataloader.collate_fn
            dataloader.collate_fn = transform_processor

        return dataloader

    def _val_dataloader(self) -> DataLoader:
        if isinstance(getattr(self, "trainer", None), pl.Trainer):
            if isinstance(self.trainer.lightning_module, flash.Task):
                self.connect(self.trainer.lightning_module)

        val_ds: Input = self._val_input
        collate_fn = self._val_dataloader_collate_fn

        transform_processor = None
        if isinstance(collate_fn, _InputTransformProcessorV2):
            transform_processor = collate_fn
            collate_fn = transform_processor.collate_fn

        if isinstance(getattr(self, "trainer", None), pl.Trainer):
            dataloader = self.trainer.lightning_module.process_val_dataset(
                val_ds,
                trainer=self.trainer,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate_fn,
            )
        else:
            dataloader = DataLoader(
                val_ds,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate_fn,
                persistent_workers=self.persistent_workers,
            )

        if transform_processor is not None:
            transform_processor.collate_fn = dataloader.collate_fn
            dataloader.collate_fn = transform_processor

        return dataloader

    def _test_dataloader(self) -> DataLoader:
        if isinstance(getattr(self, "trainer", None), pl.Trainer):
            if isinstance(self.trainer.lightning_module, flash.Task):
                self.connect(self.trainer.lightning_module)

        test_ds: Input = self._test_input
        collate_fn = self._test_dataloader_collate_fn

        transform_processor = None
        if isinstance(collate_fn, _InputTransformProcessorV2):
            transform_processor = collate_fn
            collate_fn = transform_processor.collate_fn

        if isinstance(getattr(self, "trainer", None), pl.Trainer):
            dataloader = self.trainer.lightning_module.process_test_dataset(
                test_ds,
                trainer=self.trainer,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate_fn,
            )
        else:
            dataloader = DataLoader(
                test_ds,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate_fn,
                persistent_workers=self.persistent_workers,
            )

        if transform_processor is not None:
            transform_processor.collate_fn = dataloader.collate_fn
            dataloader.collate_fn = transform_processor

        return dataloader

    def _predict_dataloader(self) -> DataLoader:
        if isinstance(getattr(self, "trainer", None), pl.Trainer):
            if isinstance(self.trainer.lightning_module, flash.Task):
                self.connect(self.trainer.lightning_module)

        predict_ds: Input = self._predict_input
        collate_fn = self._predict_dataloader_collate_fn

        transform_processor = None
        if isinstance(collate_fn, _InputTransformProcessorV2):
            transform_processor = collate_fn
            collate_fn = transform_processor.collate_fn

        if isinstance(predict_ds, IterableDataset):
            batch_size = self.batch_size
        else:
            batch_size = min(self.batch_size, len(predict_ds) if len(predict_ds) > 0 else 1)

        if isinstance(getattr(self, "trainer", None), pl.Trainer):
            dataloader = self.trainer.lightning_module.process_predict_dataset(
                predict_ds,
                batch_size=batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate_fn,
            )
        else:
            dataloader = DataLoader(
                predict_ds,
                batch_size=batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate_fn,
                persistent_workers=self.persistent_workers,
            )

        if transform_processor is not None:
            transform_processor.collate_fn = dataloader.collate_fn
            dataloader.collate_fn = transform_processor

        return dataloader

    def connect(self, task: "flash.Task"):
        data_pipeline_state = DataPipelineState()
        for properties in [
            self._train_input,
            self._val_input,
            self._test_input,
            self._predict_input,
            getattr(self._train_input, "transform", None),
            getattr(self._val_input, "transform", None),
            getattr(self._test_input, "transform", None),
            getattr(self._predict_input, "transform", None),
            task._deserializer,
            task._output_transform,
            task._output,
            task,
        ]:
            if properties is not None and hasattr(properties, "attach_data_pipeline_state"):
                properties.attach_data_pipeline_state(data_pipeline_state)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if getattr(self, "trainer", None) is None:
            return batch
        transform = None
        if self.trainer.training:
            transform = self._train_on_after_batch_transfer_fn
        elif self.trainer.validating or self.trainer.sanity_checking:
            transform = self._val_on_after_batch_transfer_fn
        elif self.trainer.testing:
            transform = self._test_on_after_batch_transfer_fn
        elif self.trainer.predicting:
            transform = self._predict_on_after_batch_transfer_fn

        if transform:
            batch = transform(batch)

        return batch

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

    @property
    def num_classes(self) -> Optional[int]:
        """Property that returns the number of classes of the datamodule if a multiclass task."""
        n_cls_train = getattr(self.train_dataset, "num_classes", None)
        n_cls_val = getattr(self.val_dataset, "num_classes", None)
        n_cls_test = getattr(self.test_dataset, "num_classes", None)
        return n_cls_train or n_cls_val or n_cls_test

    @property
    def labels(self) -> Optional[int]:
        """Property that returns the labels if this ``DataModule`` contains classification data."""
        n_cls_train = getattr(self.train_dataset, "labels", None)
        n_cls_val = getattr(self.val_dataset, "labels", None)
        n_cls_test = getattr(self.test_dataset, "labels", None)
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
        inputs = [self.train_dataset, self.val_dataset, self.test_dataset, self.predict_dataset]
        return [input for input in inputs if input]

    @property
    def input_transform(self) -> InputTransform:
        """Property that returns the input transform class used on input data."""
        # Find a better way to resolve this.
        return getattr(self.train_dataset, "transform", None) or self.input_transform_cls(RunningStage.TRAINING)

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

        if isinstance(train_dataset, IterableInput):
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
