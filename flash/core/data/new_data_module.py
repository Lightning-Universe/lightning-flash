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
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Type, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.enums import LightningEnum
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.sampler import Sampler

import flash
from flash.core.data.base_viz import BaseVisualization
from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_module import DataModule
from flash.core.data.data_pipeline import DataPipelineState
from flash.core.data.input_transform import INPUT_TRANSFORM_TYPE, InputTransform
from flash.core.data.io.input import DataKeys
from flash.core.data.io.input_base import Input
from flash.core.data.io.input_transform import DefaultInputTransform
from flash.core.data.io.output_transform import OutputTransform
from flash.core.registry import FlashRegistry
from flash.core.utilities.stages import RunningStage


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
    inputs_registry = FlashRegistry("datasets")
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
        self._data_fetcher: Optional[BaseDataFetcher] = data_fetcher or self.configure_data_fetcher()

        # TODO: Remove _X_ds reference when previous DataModule is removed.
        self._train_input = self._train_ds = train_input
        self._val_input = self._val_ds = val_input
        self._test_input = self._test_ds = test_input
        self._predict_input = self._predict_ds = predict_input

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

    @property
    def data_pipeline_state(self) -> DataPipelineState:
        """Collect the states from the input datasets and their transforms."""
        data_pipeline_state = DataPipelineState()
        if self._train_input:
            self._add_to_data_pipeline_state(self._train_input, data_pipeline_state)
        if self._val_input:
            self._add_to_data_pipeline_state(self._val_input, data_pipeline_state)
        if self._test_input:
            self._add_to_data_pipeline_state(self._test_input, data_pipeline_state)
        if self._predict_input:
            self._add_to_data_pipeline_state(self._predict_input, data_pipeline_state)
        return data_pipeline_state

    def _add_to_data_pipeline_state(self, input: Input, data_pipeline_state: DataPipelineState) -> None:
        """Collect the states contained within the input datasets and their transforms."""
        for state in input._state.values():
            data_pipeline_state.set_state(state)
        if isinstance(input.transform, InputTransform):
            for state in input.transform._state.values():
                data_pipeline_state.set_state(state)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if self.trainer.training:
            transform = self._train_on_after_batch_transfer_fn
        elif self.trainer.validating:
            transform = self._val_on_after_batch_transfer_fn
        elif self.trainer.testing:
            transform = self._test_on_after_batch_transfer_fn
        elif self.trainer.predicting:
            transform = self._predict_on_after_batch_transfer_fn

        if transform:
            batch = transform(batch)

        return batch

    @classmethod
    def create_inputs(
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
        **input_kwarg,
    ) -> Tuple[Optional[Input]]:
        cls._verify_input_enum(enum)
        input_cls: Input = cls.inputs_registry.get(enum)
        return (
            cls._create_input(
                input_cls,
                train_data,
                running_stage=RunningStage.TRAINING,
                transform=train_transform,
                **input_kwarg,
            ),
            cls._create_input(
                input_cls,
                val_data,
                running_stage=RunningStage.VALIDATING,
                transform=val_transform,
                **input_kwarg,
            ),
            cls._create_input(
                input_cls,
                test_data,
                running_stage=RunningStage.TESTING,
                transform=test_transform,
                **input_kwarg,
            ),
            cls._create_input(
                input_cls,
                predict_data,
                running_stage=RunningStage.PREDICTING,
                transform=predict_transform,
                **input_kwarg,
            ),
        )

    @staticmethod
    def _create_input(
        input_cls,
        *load_data_args,
        running_stage: RunningStage,
        transform: Optional[InputTransform],
        **kwargs,
    ) -> Optional[Input]:
        if load_data_args[0] is not None:
            return input_cls(running_stage, *load_data_args, transform=transform, **kwargs)

    @classmethod
    def _verify_input_enum(cls, enum: LightningEnum) -> None:
        if not cls.inputs_registry or not isinstance(cls.inputs_registry, FlashRegistry):
            raise MisconfigurationException(
                "The ``AutoContainer`` should have ``inputs_registry`` (FlashRegistry) populated "
                "with Input class and ``default_flash_dataset_enum`` (LightningEnum) class attributes. "
            )

        if enum not in cls.inputs_registry.available_keys():
            available_constructors = [f"from_{key.name.lower()}" for key in cls.inputs_registry.available_keys()]
            raise MisconfigurationException(
                f"The ``AutoContainer`` ``inputs_registry`` doesn't contain the associated {enum} "
                f"HINT: Here are the available constructors {available_constructors}"
            )

    @classmethod
    def register_input(cls, enum: Union[str, LightningEnum], input_cls: Type[Input]) -> None:
        if cls.inputs_registry is None:
            raise MisconfigurationException("The class attribute `inputs_registry` should be set. ")
        cls.inputs_registry(fn=input_cls, name=enum)

    @classmethod
    def from_datasets(
        cls,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        predict_dataset: Optional[Dataset] = None,
        train_transform: Optional[Union[Callable, InputTransform]] = None,
        val_transform: Optional[Union[Callable, InputTransform]] = None,
        test_transform: Optional[Union[Callable, InputTransform]] = None,
        predict_transform: Optional[Union[Callable, InputTransform]] = None,
        input_cls: Type[Input] = DatasetInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "DataModule":
        """Creates a :class:`~flash.core.data.data_module.DataModule` object from the given datasets using the
        :class:`~flash.core.data.io.input.Input`
        of name :attr:`~flash.core.data.io.input.InputFormat.DATASETS`
        from the passed or constructed :class:`~flash.core.data.io.input_transform.InputTransform`.

        Args:
            train_dataset: Dataset used during training.
            val_dataset: Dataset used during validating.
            test_dataset: Dataset used during testing.
            predict_dataset: Dataset used during predicting.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            input_cls: Input class used to create the datasets.
            transform_kwargs: Additional keyword arguments to be used when constructing the transform.
            data_module_kwargs: Additional keyword arguments to use when constructing the DataModule.

        Returns:
            The constructed data module.

        Examples::

            data_module = DataModule.from_datasets(
                train_dataset=train_dataset,
                train_transform={
                    "per_sample_transform": torch.as_tensor,
                },
            )
        """
        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_dataset, transform=train_transform, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_dataset, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_dataset, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_dataset, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )
