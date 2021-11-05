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
from typing import Any, Optional, Tuple, Type, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.enums import LightningEnum
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.utils.data.sampler import Sampler

import flash
from flash.core.data.base_viz import BaseVisualization
from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_module import DataModule
from flash.core.data.data_pipeline import DefaultPreprocess, Postprocess
from flash.core.data.datasets import BaseDataset
from flash.core.data.input_transform import INPUT_TRANSFORM_TYPE, InputTransform
from flash.core.registry import FlashRegistry
from flash.core.utilities.stages import RunningStage


class DataModule(DataModule):
    """A basic DataModule class for all Flash tasks. This class includes references to a
    :class:`~flash.core.data.datasets.BaseDataset` and a :class:`~flash.core.data.callback.BaseDataFetcher`.

    Args:
        train_dataset: Dataset for training. Defaults to None.
        val_dataset: Dataset for validating model performance during training. Defaults to None.
        test_dataset: Dataset to test model performance. Defaults to None.
        predict_dataset: Dataset for predicting. Defaults to None.
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
    flash_datasets_registry = FlashRegistry("datasets")

    def __init__(
        self,
        train_dataset: Optional[BaseDataset] = None,
        val_dataset: Optional[BaseDataset] = None,
        test_dataset: Optional[BaseDataset] = None,
        predict_dataset: Optional[BaseDataset] = None,
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

        self._postprocess: Optional[Postprocess] = None
        self._viz: Optional[BaseVisualization] = None
        self._data_fetcher: Optional[BaseDataFetcher] = data_fetcher or self.configure_data_fetcher()

        self._train_ds = train_dataset
        self._val_ds = val_dataset
        self._test_ds = test_dataset
        self._predict_ds = predict_dataset

        if self._train_ds and self._val_ds and isinstance(val_split, float) and val_split > 0:
            raise MisconfigurationException(
                "A `val_dataset` was provided with `val_split`. Please, choose one or the other."
            )

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

        if num_workers is None:
            num_workers = 0
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers and num_workers > 0
        self.pin_memory = pin_memory

        self.sampler = sampler

        self.set_running_stages()

        LightningDataModule.__init__(self)

    def _train_dataloader(self) -> DataLoader:
        train_ds: BaseDataset = self._train_ds
        collate_fn = train_ds.dataloader_collate_fn
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
        val_ds: BaseDataset = self._val_ds
        collate_fn = val_ds.dataloader_collate_fn

        if isinstance(getattr(self, "trainer", None), pl.Trainer):
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
        test_ds: BaseDataset = self._test_ds
        collate_fn = test_ds.dataloader_collate_fn

        if isinstance(getattr(self, "trainer", None), pl.Trainer):
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
        predict_ds: BaseDataset = self._predict_ds
        collate_fn = predict_ds.dataloader_collate_fn

        if isinstance(predict_ds, IterableDataset):
            batch_size = self.batch_size
        else:
            batch_size = min(self.batch_size, len(predict_ds) if len(predict_ds) > 0 else 1)

        if isinstance(getattr(self, "trainer", None), pl.Trainer):
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

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        ds = None
        if self.trainer.training:
            ds = self._train_ds
        elif self.trainer.validating:
            ds = self._val_ds
        elif self.trainer.testing:
            ds = self._test_ds
        elif self.trainer.predicting:
            ds = self._predict_ds

        if ds:
            transform = ds.on_after_batch_transfer_fn
            batch = transform(batch)

        return batch

    @classmethod
    def create_flash_datasets(
        cls,
        enum: Union[LightningEnum, str],
        train_data: Optional[Any] = None,
        val_data: Optional[Any] = None,
        test_data: Optional[Any] = None,
        predict_data: Optional[Any] = None,
        train_transform: Optional[INPUT_TRANSFORM_TYPE] = None,
        val_transform: Optional[INPUT_TRANSFORM_TYPE] = None,
        test_transform: Optional[INPUT_TRANSFORM_TYPE] = None,
        predict_transform: Optional[INPUT_TRANSFORM_TYPE] = None,
        **flash_dataset_kwargs,
    ) -> Tuple[Optional[BaseDataset]]:
        cls._verify_flash_dataset_enum(enum)
        flash_dataset_cls: BaseDataset = cls.flash_datasets_registry.get(enum)
        return (
            cls._create_flash_dataset(
                flash_dataset_cls,
                train_data,
                running_stage=RunningStage.TRAINING,
                transform=train_transform,
                **flash_dataset_kwargs,
            ),
            cls._create_flash_dataset(
                flash_dataset_cls,
                val_data,
                running_stage=RunningStage.VALIDATING,
                transform=val_transform,
                **flash_dataset_kwargs,
            ),
            cls._create_flash_dataset(
                flash_dataset_cls,
                test_data,
                running_stage=RunningStage.TESTING,
                transform=test_transform,
                **flash_dataset_kwargs,
            ),
            cls._create_flash_dataset(
                flash_dataset_cls,
                predict_data,
                running_stage=RunningStage.PREDICTING,
                transform=predict_transform,
                **flash_dataset_kwargs,
            ),
        )

    @staticmethod
    def _create_flash_dataset(
        flash_dataset_cls,
        *load_data_args,
        running_stage: RunningStage,
        transform: Optional[InputTransform],
        **kwargs,
    ) -> Optional[BaseDataset]:
        if load_data_args[0] is not None:
            return flash_dataset_cls.from_data(
                *load_data_args, running_stage=running_stage, transform=transform, **kwargs
            )

    @classmethod
    def _verify_flash_dataset_enum(cls, enum: LightningEnum) -> None:
        if not cls.flash_datasets_registry or not isinstance(cls.flash_datasets_registry, FlashRegistry):
            raise MisconfigurationException(
                "The ``AutoContainer`` should have ``flash_datasets_registry`` (FlashRegistry) populated "
                "with datasource class and ``default_flash_dataset_enum`` (LightningEnum) class attributes. "
            )

        if enum not in cls.flash_datasets_registry.available_keys():
            available_constructors = [
                f"from_{key.name.lower()}" for key in cls.flash_datasets_registry.available_keys()
            ]
            raise MisconfigurationException(
                f"The ``AutoContainer`` ``flash_datasets_registry`` doesn't contain the associated {enum} "
                f"HINT: Here are the available constructors {available_constructors}"
            )

    @classmethod
    def register_flash_dataset(cls, enum: Union[str, LightningEnum], flash_dataset_cls: Type[BaseDataset]) -> None:
        if cls.flash_datasets_registry is None:
            raise MisconfigurationException("The class attribute `flash_datasets_registry` should be set. ")
        cls.flash_datasets_registry(fn=flash_dataset_cls, name=enum)
