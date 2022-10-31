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
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import IterableDataset
from torch.utils.data.sampler import Sampler

import flash
from flash.core.data.base_viz import BaseVisualization
from flash.core.data.callback import BaseDataFetcher
from flash.core.data.io.input import DataKeys, Input, IterableInput
from flash.core.data.io.input_transform import (
    create_device_input_transform_processor,
    create_or_configure_input_transform,
    create_worker_input_transform_processor,
    InputTransform,
)
from flash.core.data.splits import SplitDataset
from flash.core.data.utils import _STAGES_PREFIX
from flash.core.utilities.imports import _CORE_TESTING
from flash.core.utilities.stages import RunningStage
from flash.core.utilities.types import INPUT_TRANSFORM_TYPE

# Skip doctests if requirements aren't available
if not _CORE_TESTING:
    __doctest_skip__ = ["DataModule"]


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
        transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
        transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
        val_split: An optional float which gives the relative amount of the training dataset to use for the validation
            dataset.
        batch_size: The batch size to be used by the DataLoader.
        num_workers: The number of workers to use for parallelized loading.
        sampler: A sampler following the :class:`~torch.utils.data.sampler.Sampler` type.
            Will be passed to the DataLoader for the training dataset. Defaults to None.

    Examples
    ________

    .. testsetup::

        >>> from flash import DataModule
        >>> from flash.core.utilities.stages import RunningStage
        >>> from torch.utils.data.sampler import SequentialSampler, WeightedRandomSampler
        >>> class TestInput(Input):
        ...     def train_load_data(self, _):
        ...         return [(0, 1, 2, 3), (0, 1, 2, 3)]
        >>> train_input = TestInput(RunningStage.TRAINING, [1])

    You can provide the sampler to use for the train dataloader using the ``sampler`` argument.
    The sampler can be a function or type that needs the dataset as an argument:

    .. doctest::

        >>> datamodule = DataModule(train_input, sampler=SequentialSampler, batch_size=1)
        >>> print(datamodule.train_dataloader().sampler)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        <torch.utils.data.sampler.SequentialSampler object at ...>

    Alternatively, you can pass a sampler instance:

    .. doctest::

        >>> datamodule = DataModule(train_input, sampler=WeightedRandomSampler([0.1, 0.5], 2), batch_size=1)
        >>> print(datamodule.train_dataloader().sampler)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        <torch.utils.data.sampler.WeightedRandomSampler object at ...>

    """

    input_transform_cls = InputTransform

    def __init__(
        self,
        train_input: Optional[Input] = None,
        val_input: Optional[Input] = None,
        test_input: Optional[Input] = None,
        predict_input: Optional[Input] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        transform: INPUT_TRANSFORM_TYPE = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        val_split: Optional[float] = None,
        batch_size: Optional[int] = None,
        num_workers: int = 0,
        sampler: Optional[Union[Callable, Sampler, Type[Sampler]]] = None,
        pin_memory: bool = True,
        persistent_workers: bool = False,
    ) -> None:

        if not batch_size:
            raise TypeError("The `batch_size` should be provided to the DataModule on instantiation.")

        if flash._IS_TESTING and torch.cuda.is_available():
            batch_size = 16

        self.input_transform = create_or_configure_input_transform(
            transform=transform, transform_kwargs=transform_kwargs
        )

        self.viz: Optional[BaseVisualization] = None

        self._train_input = train_input
        self._val_input = val_input
        self._test_input = test_input
        self._predict_input = predict_input

        if self._train_input and self._val_input and isinstance(val_split, float) and val_split > 0:
            raise TypeError("A `val_dataset` was provided with `val_split`. Please, choose one or the other.")

        if self._train_input and (val_split is not None and not self._val_input):
            self._train_input, self._val_input = self._split_train_val(self._train_input, val_split)

        self._data_fetcher: Optional[BaseDataFetcher] = data_fetcher or self.configure_data_fetcher()

        self._on_after_batch_transfer_fns = None

        if self._train_input:
            self.train_dataloader = self._train_dataloader

        if self._val_input:
            self.val_dataloader = self._val_dataloader

        if self._test_input:
            self.test_dataloader = self._test_dataloader

        if self._predict_input:
            self.predict_dataloader = self._predict_dataloader

        self.batch_size = batch_size

        self.num_workers = num_workers
        self.persistent_workers = persistent_workers and num_workers > 0
        self.pin_memory = pin_memory

        self.sampler = sampler

        super().__init__()

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
        """This property returns the prediction dataset."""
        return self._predict_input

    #####################################
    # METHODS PERTAINING TO DATALOADERS #
    #####################################

    def _resolve_input_transform(self) -> Optional[InputTransform]:
        input_transform = self.input_transform
        if (
            isinstance(getattr(self, "trainer", None), pl.Trainer)
            and getattr(self.trainer.lightning_module, "input_transform", None) is not None
        ):
            input_transform = create_or_configure_input_transform(self.trainer.lightning_module.input_transform)

        if input_transform is not None:
            input_transform.callbacks = [self.data_fetcher]
        return input_transform

    def _train_dataloader(self) -> DataLoader:
        train_ds: Input = self._train_input

        input_transform = self._resolve_input_transform()

        shuffle: bool = False
        if isinstance(train_ds, IterableDataset):
            drop_last = False
        else:
            drop_last = len(train_ds) > self.batch_size

        if self.sampler is None:
            sampler = None
            shuffle = not isinstance(train_ds, IterableDataset)
        elif callable(self.sampler):
            sampler = self.sampler(train_ds)
        else:
            sampler = self.sampler

        if isinstance(getattr(self, "trainer", None), pl.Trainer) and hasattr(
            self.trainer.lightning_module, "process_train_dataset"
        ):
            dataloader = self.trainer.lightning_module.process_train_dataset(
                train_ds,
                self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=shuffle,
                drop_last=drop_last,
                sampler=sampler,
                persistent_workers=self.persistent_workers,
                input_transform=input_transform,
                trainer=self.trainer,
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
                collate_fn=create_worker_input_transform_processor(RunningStage.TRAINING, input_transform),
                persistent_workers=self.persistent_workers,
            )

        self._on_after_batch_transfer_fns = None
        return dataloader

    def _val_dataloader(self) -> DataLoader:
        val_ds: Input = self._val_input

        input_transform = self._resolve_input_transform()

        if isinstance(getattr(self, "trainer", None), pl.Trainer) and hasattr(
            self.trainer.lightning_module, "process_val_dataset"
        ):
            dataloader = self.trainer.lightning_module.process_val_dataset(
                val_ds,
                self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                input_transform=input_transform,
                trainer=self.trainer,
            )
        else:
            dataloader = DataLoader(
                val_ds,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=create_worker_input_transform_processor(RunningStage.VALIDATING, input_transform),
                persistent_workers=self.persistent_workers,
            )

        self._on_after_batch_transfer_fns = None
        return dataloader

    def _test_dataloader(self) -> DataLoader:
        test_ds: Input = self._test_input

        input_transform = self._resolve_input_transform()

        if isinstance(getattr(self, "trainer", None), pl.Trainer) and hasattr(
            self.trainer.lightning_module, "process_test_dataset"
        ):
            dataloader = self.trainer.lightning_module.process_test_dataset(
                test_ds,
                self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                input_transform=input_transform,
                trainer=self.trainer,
            )
        else:
            dataloader = DataLoader(
                test_ds,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=create_worker_input_transform_processor(RunningStage.TESTING, input_transform),
                persistent_workers=self.persistent_workers,
            )

        self._on_after_batch_transfer_fns = None
        return dataloader

    def _predict_dataloader(self) -> DataLoader:
        predict_ds: Input = self._predict_input

        input_transform = self._resolve_input_transform()

        if isinstance(predict_ds, IterableDataset):
            batch_size = self.batch_size
        else:
            batch_size = min(self.batch_size, len(predict_ds) if len(predict_ds) > 0 else 1)

        if isinstance(getattr(self, "trainer", None), pl.Trainer) and hasattr(
            self.trainer.lightning_module, "process_predict_dataset"
        ):
            dataloader = self.trainer.lightning_module.process_predict_dataset(
                predict_ds,
                self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                input_transform=input_transform,
                trainer=self.trainer,
            )
        else:
            dataloader = DataLoader(
                predict_ds,
                batch_size=batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=create_worker_input_transform_processor(RunningStage.PREDICTING, input_transform),
                persistent_workers=self.persistent_workers,
            )

        self._on_after_batch_transfer_fns = None
        return dataloader

    ############################################################
    # METHODS RELATED TO on_after_batch_transfer FUNCTIONALITY #
    ############################################################

    def _load_on_after_batch_transfer_fns(self) -> None:
        self._on_after_batch_transfer_fns = {}

        for stage in [
            RunningStage.TRAINING,
            RunningStage.VALIDATING,
            RunningStage.SANITY_CHECKING,
            RunningStage.TESTING,
            RunningStage.PREDICTING,
        ]:
            input_transform = self._resolve_input_transform()

            if input_transform is not None:
                transform = create_device_input_transform_processor(
                    stage if stage != RunningStage.SANITY_CHECKING else RunningStage.VALIDATING,
                    input_transform,
                )
                self._on_after_batch_transfer_fns[stage] = transform
            else:
                self._on_after_batch_transfer_fns[stage] = None

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if getattr(self, "trainer", None) is None:
            return batch

        if self._on_after_batch_transfer_fns is None:
            self._load_on_after_batch_transfer_fns()

        stage = self.trainer.state.stage

        transform = self._on_after_batch_transfer_fns[stage]

        if transform:
            batch = transform(batch)
        return batch

    ###################################
    # METHODS RELATED TO DATA FETCHER #
    ###################################

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

    ######################################
    # METHODS RELATED TO INPUT TRANSFORM #
    ######################################

    @property
    def input_transform(self) -> InputTransform:
        """This property returns the data fetcher."""
        return self._input_transform

    @input_transform.setter
    def input_transform(self, input_transform: InputTransform) -> None:
        self._input_transform = input_transform

    ####################################
    # METHODS RELATED TO VISUALIZATION #
    ####################################

    @property
    def viz(self) -> BaseVisualization:
        return self._viz or DataModule.configure_data_fetcher()

    @viz.setter
    def viz(self, viz: BaseVisualization) -> None:
        self._viz = viz

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

    def _show_batch(
        self,
        stage: str,
        func_names: Union[str, List[str]],
        limit_nb_samples: int = None,
        figsize: Tuple[int, int] = (6.4, 4.8),
        reset: bool = True,
    ) -> None:
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

        if not limit_nb_samples:
            limit_nb_samples = self.batch_size

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
            data_fetcher._show(stage, func_names, limit_nb_samples, figsize)
            if reset:
                self.data_fetcher.batches[stage] = {}

    def show_train_batch(
        self,
        hooks_names: Union[str, List[str]] = "load_sample",
        reset: bool = True,
        limit_nb_samples: int = None,
        figsize: Tuple[int, int] = (6.4, 4.8),
    ) -> None:
        """This function is used to visualize a batch from the train dataloader."""
        stage_name: str = _STAGES_PREFIX[RunningStage.TRAINING]
        self._show_batch(stage_name, hooks_names, limit_nb_samples, figsize, reset=reset)

    def show_val_batch(
        self,
        hooks_names: Union[str, List[str]] = "load_sample",
        reset: bool = True,
        limit_nb_samples: int = None,
        figsize: Tuple[int, int] = (6.4, 4.8),
    ) -> None:
        """This function is used to visualize a batch from the validation dataloader."""
        stage_name: str = _STAGES_PREFIX[RunningStage.VALIDATING]
        self._show_batch(stage_name, hooks_names, limit_nb_samples, figsize, reset=reset)

    def show_test_batch(
        self,
        hooks_names: Union[str, List[str]] = "load_sample",
        reset: bool = True,
        limit_nb_samples: int = None,
        figsize: Tuple[int, int] = (6.4, 4.8),
    ) -> None:
        """This function is used to visualize a batch from the test dataloader."""
        stage_name: str = _STAGES_PREFIX[RunningStage.TESTING]
        self._show_batch(stage_name, hooks_names, limit_nb_samples, figsize, reset=reset)

    def show_predict_batch(
        self,
        hooks_names: Union[str, List[str]] = "load_sample",
        reset: bool = True,
        limit_nb_samples: int = None,
        figsize: Tuple[int, int] = (6.4, 4.8),
    ) -> None:
        """This function is used to visualize a batch from the prediction dataloader."""
        stage_name: str = _STAGES_PREFIX[RunningStage.PREDICTING]
        self._show_batch(stage_name, hooks_names, limit_nb_samples, figsize, reset=reset)

    def _get_property(self, property_name: str) -> Optional[Any]:
        train = getattr(self.train_dataset, property_name, None)
        val = getattr(self.val_dataset, property_name, None)
        test = getattr(self.test_dataset, property_name, None)
        filtered = list(filter(lambda x: x is not None, [train, val, test]))
        return filtered[0] if len(filtered) > 0 else None

    @property
    def num_classes(self) -> Optional[int]:
        """Property that returns the number of classes of the datamodule if a multiclass task."""
        return self._get_property("num_classes")

    @property
    def labels(self) -> Optional[int]:
        """Property that returns the labels if this ``DataModule`` contains classification data."""
        return self._get_property("labels")

    @property
    def multi_label(self) -> Optional[bool]:
        """Property that returns ``True`` if this ``DataModule`` contains multi-label data."""
        return self._get_property("multi_label")

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
            raise ValueError(f"`val_split` should be a float between 0 and 1. Found {val_split}.")

        if isinstance(train_dataset, IterableInput):
            raise ValueError("`val_split` should be `None` when the dataset is built with an IterableDataset.")

        val_num_samples = int(len(train_dataset) * val_split)
        indices = list(range(len(train_dataset)))
        np.random.shuffle(indices)
        val_indices = indices[:val_num_samples]
        train_indices = indices[val_num_samples:]
        return (
            SplitDataset(
                train_dataset, train_indices, running_stage=RunningStage.TRAINING, use_duplicated_indices=True
            ),
            SplitDataset(
                train_dataset, val_indices, running_stage=RunningStage.VALIDATING, use_duplicated_indices=True
            ),
        )
