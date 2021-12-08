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
from typing import Any, Callable, Mapping, Optional, Type

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataset import IterableDataset
from torch.utils.data.sampler import Sampler

import flash
from flash.core.data.base_viz import BaseVisualization
from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_module import DataModule
from flash.core.data.data_pipeline import DataPipelineState
from flash.core.data.input_transform import InputTransform
from flash.core.data.io.input import DataKeys
from flash.core.data.io.input_base import Input
from flash.core.data.io.input_transform import DefaultInputTransform
from flash.core.data.io.output_transform import OutputTransform
from flash.core.registry import FlashRegistry


class DatasetInput(Input):
    """The ``DatasetInput`` implements default behaviours for data sources which expect the input to
    :meth:`~flash.core.data.io.input.Input.load_data` to be a :class:`torch.utils.data.dataset.Dataset`

    Args:
        labels: Optionally pass the labels as a mapping from class index to label string. These will then be set as the
            :class:`~flash.core.data.io.input.ClassificationState`.
    """

    def load_sample(self, sample: Any) -> Mapping[str, Any]:
        if isinstance(sample, tuple) and len(sample) == 2:
            return {DataKeys.INPUT: sample[0], DataKeys.TARGET: sample[1]}
        return {DataKeys.INPUT: sample}


class DataModule(DataModule):
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

    input_transform_cls = DefaultInputTransform
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
    ) -> None:

        if not batch_size:
            raise MisconfigurationException("The `batch_size` should be provided to the DataModule on instantiation.")

        if flash._IS_TESTING and torch.cuda.is_available():
            batch_size = 16

        self._input_transform: Optional[OutputTransform] = None
        self._output_transform: Optional[OutputTransform] = None
        self._viz: Optional[BaseVisualization] = None

        # TODO: Remove _X_ds reference when previous DataModule is removed.
        self._train_input = self._train_ds = train_input
        self._val_input = self._val_ds = val_input
        self._test_input = self._test_ds = test_input
        self._predict_input = self._predict_ds = predict_input

        self._data_fetcher: Optional[BaseDataFetcher] = data_fetcher or self.configure_data_fetcher()

        self._train_dataloader_collate_fn = self._resolve_dataloader_collate_fn(self._train_input)
        self._val_dataloader_collate_fn = self._resolve_dataloader_collate_fn(self._val_input)
        self._test_dataloader_collate_fn = self._resolve_dataloader_collate_fn(self._test_input)
        self._predict_dataloader_collate_fn = self._resolve_dataloader_collate_fn(self._predict_input)

        self._train_on_after_batch_transfer_fn = self._resolve_on_after_batch_transfer_fn(self._train_input)
        self._val_on_after_batch_transfer_fn = self._resolve_on_after_batch_transfer_fn(self._val_input)
        self._test_on_after_batch_transfer_fn = self._resolve_on_after_batch_transfer_fn(self._test_input)
        self._predict_on_after_batch_transfer_fn = self._resolve_on_after_batch_transfer_fn(self._predict_input)

        if self._train_input and self._val_input and isinstance(val_split, float) and val_split > 0:
            raise MisconfigurationException(
                "A `val_dataset` was provided with `val_split`. Please, choose one or the other."
            )

        if self._train_input is not None and (val_split is not None and self._val_input is None):
            self._train_input, self._val_input = self._split_train_val(self._train_input, val_split)

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

        LightningDataModule.__init__(self)

    @property
    def input_transform(self) -> InputTransform:
        """Property that returns the input transform class used on input data."""
        # Find a better way to resolve this.
        return self._train_ds.transform or self.input_transform_cls()

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
        train_ds: Input = self._train_input
        collate_fn = self._train_dataloader_collate_fn
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
            if isinstance(self.trainer.lightning_module, flash.Task):
                self.connect(self.trainer.lightning_module)
            return self.trainer.lightning_module.process_train_dataset(
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

        return DataLoader(
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

    def _val_dataloader(self) -> DataLoader:
        val_ds: Input = self._val_input
        collate_fn = self._val_dataloader_collate_fn

        if isinstance(getattr(self, "trainer", None), pl.Trainer):
            if isinstance(self.trainer.lightning_module, flash.Task):
                self.connect(self.trainer.lightning_module)
            return self.trainer.lightning_module.process_val_dataset(
                val_ds,
                trainer=self.trainer,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate_fn,
            )

        return DataLoader(
            val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers,
        )

    def _test_dataloader(self) -> DataLoader:
        test_ds: Input = self._test_input
        collate_fn = self._test_dataloader_collate_fn

        if isinstance(getattr(self, "trainer", None), pl.Trainer):
            if isinstance(self.trainer.lightning_module, flash.Task):
                self.connect(self.trainer.lightning_module)
            return self.trainer.lightning_module.process_test_dataset(
                test_ds,
                trainer=self.trainer,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate_fn,
            )

        return DataLoader(
            test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers,
        )

    def _predict_dataloader(self) -> DataLoader:
        predict_ds: Input = self._predict_input
        collate_fn = self._predict_dataloader_collate_fn

        if isinstance(predict_ds, IterableDataset):
            batch_size = self.batch_size
        else:
            batch_size = min(self.batch_size, len(predict_ds) if len(predict_ds) > 0 else 1)

        if isinstance(getattr(self, "trainer", None), pl.Trainer):
            if isinstance(self.trainer.lightning_module, flash.Task):
                self.connect(self.trainer.lightning_module)
            return self.trainer.lightning_module.process_predict_dataset(
                predict_ds,
                batch_size=batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate_fn,
            )

        return DataLoader(
            predict_ds,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers,
        )

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
            if properties is not None:
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
