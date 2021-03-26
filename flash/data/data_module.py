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
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.core.datamodule import _DataModuleWrapper
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Subset

from flash.data.auto_dataset import AutoDataset
from flash.data.data_pipeline import DataPipeline, Postprocess, Preprocess


class DataModule(pl.LightningDataModule):
    """Basic DataModule class for all Flash tasks

    Args:
        train_ds: Dataset for training. Defaults to None.
        valid_ds: Dataset for VALIDATING model performance during training. Defaults to None.
        test_ds: Dataset to test model performance. Defaults to None.
        predict_ds: Dataset for predicting. Defaults to None.
        batch_size: the batch size to be used by the DataLoader. Defaults to 1.
        num_workers: The number of workers to use for parallelized loading.
            Defaults to None which equals the number of available CPU threads,
            or 0 for Darwin platform.
    """

    preprocess_cls = Preprocess
    postprocess_cls = Postprocess

    def __init__(
        self,
        train_ds: Optional[Dataset] = None,
        valid_ds: Optional[Dataset] = None,
        test_ds: Optional[Dataset] = None,
        predict_ds: Optional[Dataset] = None,
        batch_size: int = 1,
        num_workers: Optional[int] = None,
    ) -> None:

        super().__init__()
        self._train_ds = train_ds
        self._valid_ds = valid_ds
        self._test_ds = test_ds
        self._predict_ds = predict_ds

        if self._train_ds:
            self.train_dataloader = self._train_dataloader

        if self._valid_ds:
            self.val_dataloader = self._val_dataloader

        if self._test_ds:
            self.test_dataloader = self._test_dataloader

        if self._predict_ds:
            self.predict_dataloader = self._predict_dataloader

        self.batch_size = batch_size

        # TODO: figure out best solution for setting num_workers
        if num_workers is None:
            num_workers = 0 if platform.system() == "Darwin" else os.cpu_count()
        self.num_workers = num_workers

        self._preprocess = None
        self._postprocess = None

        # this may also trigger data preloading
        self.set_running_stages()

    @staticmethod
    def get_dataset_attribute(dataset: torch.utils.data.Dataset, attr_name: str, default: Optional[Any] = None) -> Any:
        if isinstance(dataset, Subset):
            return getattr(dataset.dataset, attr_name, default)

        return getattr(dataset, attr_name, default)

    @staticmethod
    def set_dataset_attribute(dataset: torch.utils.data.Dataset, attr_name: str, value: Any) -> None:
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
        setattr(dataset, attr_name, value)

    def set_running_stages(self):
        if self._train_ds:
            self.set_dataset_attribute(self._train_ds, 'running_stage', RunningStage.TRAINING)

        if self._valid_ds:
            self.set_dataset_attribute(self._valid_ds, 'running_stage', RunningStage.VALIDATING)

        if self._test_ds:
            self.set_dataset_attribute(self._test_ds, 'running_stage', RunningStage.TESTING)

        if self._predict_ds:
            self.set_dataset_attribute(self._predict_ds, 'running_stage', RunningStage.PREDICTING)

    def _resolve_collate_fn(self, dataset: Dataset, running_stage: RunningStage) -> Optional[Callable]:
        if isinstance(dataset, AutoDataset):
            return self.data_pipeline.worker_preprocessor(running_stage)

    def _train_dataloader(self) -> DataLoader:
        train_ds: Dataset = self._train_ds() if isinstance(self._train_ds, Callable) else self._train_ds
        return DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self._resolve_collate_fn(train_ds, RunningStage.TRAINING)
        )

    def _val_dataloader(self) -> DataLoader:
        valid_ds: Dataset = self._valid_ds() if isinstance(self._valid_ds, Callable) else self._valid_ds
        return DataLoader(
            valid_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._resolve_collate_fn(valid_ds, RunningStage.VALIDATING)
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
        return DataLoader(
            predict_ds,
            batch_size=min(self.batch_size,
                           len(predict_ds) if len(predict_ds) > 0 else 1),
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._resolve_collate_fn(predict_ds, RunningStage.PREDICTING)
        )

    def generate_auto_dataset(self, *args, **kwargs):
        if all(a is None for a in args) and len(kwargs) == 0:
            return None
        return self.data_pipeline._generate_auto_dataset(*args, **kwargs)

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
    ) -> AutoDataset:
        """
        This function is used to generate an ``AutoDataset`` from a ``DataPipeline`` if provided
        or from the provided ``whole_data_load_fn``, ``per_sample_load_fn`` functions directly
        """

        if whole_data_load_fn is None:
            whole_data_load_fn = getattr(
                cls.preprocess_cls,
                DataPipeline._resolve_function_hierarchy('load_data', cls.preprocess_cls, running_stage, Preprocess)
            )

        if per_sample_load_fn is None:
            per_sample_load_fn = getattr(
                cls.preprocess_cls,
                DataPipeline._resolve_function_hierarchy('load_sample', cls.preprocess_cls, running_stage, Preprocess)
            )
        return AutoDataset(data, whole_data_load_fn, per_sample_load_fn, data_pipeline, running_stage=running_stage)

    @staticmethod
    def train_valid_test_split(
        dataset: torch.utils.data.Dataset,
        train_split: Optional[Union[float, int]] = None,
        valid_split: Optional[Union[float, int]] = None,
        test_split: Optional[Union[float, int]] = None,
        seed: Optional[int] = 1234,
    ) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
        """Returns split Datasets based on train, valid & test split parameters

        Args:
            dataset: Dataset to be split.
            train_split: If Float, ratio of data to be contained within the train dataset. If Int,
                number of samples to be contained within train dataset.
            valid_split: If Float, ratio of data to be contained within the validation dataset. If Int,
                number of samples to be contained within test dataset.
            test_split: If Float, ratio of data to be contained within the test dataset. If Int,
                number of samples to be contained within test dataset.
            seed: Used for the train/val splits when valid_split is not None.

        """
        n = len(dataset)

        if test_split is None:
            _test_length = 0
        elif isinstance(test_split, float):
            _test_length = int(n * test_split)
        else:
            _test_length = test_split

        if valid_split is None:
            _val_length = 0
        elif isinstance(valid_split, float):
            _val_length = int(n * valid_split)
        else:
            _val_length = valid_split

        if train_split is None:
            _train_length = n - _val_length - _test_length
        elif isinstance(train_split, float):
            _train_length = int(n * train_split)
        else:
            _train_length = train_split

        if seed:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None

        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            dataset, [_train_length, _val_length, _test_length], generator
        )
        if valid_split is None:
            val_ds = None
        if test_split is None:
            test_ds = None

        return train_ds, val_ds, test_ds

    @classmethod
    def _generate_dataset_if_possible(
        cls,
        data: Optional[Any],
        running_stage: RunningStage,
        whole_data_load_fn: Optional[Callable] = None,
        per_sample_load_fn: Optional[Callable] = None,
        data_pipeline: Optional[DataPipeline] = None
    ) -> Optional[AutoDataset]:
        if data is None:
            return

        if data_pipeline:
            return data_pipeline._generate_auto_dataset(data, running_stage=running_stage)

        return cls.autogenerate_dataset(data, running_stage, whole_data_load_fn, per_sample_load_fn, data_pipeline)

    @classmethod
    def from_load_data_inputs(
        cls,
        train_load_data_input: Optional[Any] = None,
        valid_load_data_input: Optional[Any] = None,
        test_load_data_input: Optional[Any] = None,
        predict_load_data_input: Optional[Any] = None,
        preprocess: Optional[Preprocess] = None,
        postprocess: Optional[Postprocess] = None,
        **kwargs,
    ) -> 'DataModule':
        """
        This functions is an helper to generate a ``DataModule`` from a ``DataPipeline``.

        Args:
            cls: ``DataModule`` subclass
            train_load_data_input: Data to be received by the ``train_load_data`` function from this ``Preprocess``
            valid_load_data_input: Data to be received by the ``val_load_data`` function from this ``Preprocess``
            test_load_data_input: Data to be received by the ``test_load_data`` function from this ``Preprocess``
            predict_load_data_input: Data to be received by the ``predict_load_data`` function from this ``Preprocess``
            kwargs: Any extra arguments to instantiate the provided ``DataModule``
        """
        # trick to get data_pipeline from empty DataModule
        if preprocess is not None or postprocess is not None:
            data_pipeline = DataPipeline(
                preprocess or cls(**kwargs).preprocess, postprocess or cls(**kwargs).postprocess
            )
        else:
            data_pipeline = cls(**kwargs).data_pipeline
        train_ds = cls._generate_dataset_if_possible(
            train_load_data_input, running_stage=RunningStage.TRAINING, data_pipeline=data_pipeline
        )
        valid_ds = cls._generate_dataset_if_possible(
            valid_load_data_input, running_stage=RunningStage.VALIDATING, data_pipeline=data_pipeline
        )
        test_ds = cls._generate_dataset_if_possible(
            test_load_data_input, running_stage=RunningStage.TESTING, data_pipeline=data_pipeline
        )
        predict_ds = cls._generate_dataset_if_possible(
            predict_load_data_input, running_stage=RunningStage.PREDICTING, data_pipeline=data_pipeline
        )
        datamodule = cls(train_ds=train_ds, valid_ds=valid_ds, test_ds=test_ds, predict_ds=predict_ds, **kwargs)
        datamodule._preprocess = data_pipeline._preprocess_pipeline
        datamodule._postprocess = data_pipeline._postprocess_pipeline
        return datamodule
