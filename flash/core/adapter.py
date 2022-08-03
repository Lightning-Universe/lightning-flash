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
from abc import abstractmethod
from typing import Any, Callable, Optional

import torch.jit
from torch import nn, Tensor
from torch.utils.data import DataLoader, Sampler

import flash
from flash.core.data.io.input import InputBase
from flash.core.data.io.input_transform import InputTransform
from flash.core.model import DatasetProcessor, ModuleWrapperBase, Task
from flash.core.utilities.types import INPUT_TRANSFORM_TYPE


class Adapter(DatasetProcessor, ModuleWrapperBase, nn.Module):
    """The ``Adapter`` is a lightweight interface that can be used to encapsulate the logic from a particular
    provider within a :class:`~flash.core.model.Task`."""

    @classmethod
    @abstractmethod
    def from_task(cls, task: "flash.Task", **kwargs) -> "Adapter":
        """Instantiate the adapter from the given :class:`~flash.core.model.Task`.

        This includes resolution / creation of backbones / heads and any other provider specific options.
        """

    def forward(self, x: Any) -> Any:
        pass

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        pass

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        pass

    def test_step(self, batch: Any, batch_idx: int) -> None:
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        pass

    def training_epoch_end(self, outputs) -> None:
        pass

    def validation_epoch_end(self, outputs) -> None:
        pass

    def test_epoch_end(self, outputs) -> None:
        pass


def identity_collate_fn(x):
    return x


class AdapterTask(Task):
    """The ``AdapterTask`` is a :class:`~flash.core.model.Task` which wraps an :class:`~flash.core.adapter.Adapter`
    and forwards all of the hooks.

    Args:
        adapter: The :class:`~flash.core.adapter.Adapter` to wrap.
        kwargs: Keyword arguments to be passed to the base :class:`~flash.core.model.Task`.
    """

    def __init__(self, adapter: Adapter, **kwargs):
        super().__init__(**kwargs)

        self.adapter = adapter

    @torch.jit.unused
    @property
    def input_transform(self) -> Optional[INPUT_TRANSFORM_TYPE]:
        return self.adapter.input_transform

    @input_transform.setter
    def input_transform(self, input_transform: INPUT_TRANSFORM_TYPE) -> None:
        self.adapter.input_transform = input_transform

    @torch.jit.unused
    @property
    def collate_fn(self) -> Optional[Callable]:
        return self.adapter.collate_fn

    @collate_fn.setter
    def collate_fn(self, collate_fn: Callable) -> None:
        self.adapter.collate_fn = collate_fn

    @torch.jit.unused
    @property
    def backbone(self) -> nn.Module:
        return self.adapter.backbone

    def forward(self, x: Tensor) -> Any:
        return self.adapter.forward(x)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        return self.adapter.training_step(batch, batch_idx)

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        return self.adapter.validation_step(batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        return self.adapter.test_step(batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self.adapter.predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)

    def training_epoch_end(self, outputs) -> None:
        return self.adapter.training_epoch_end(outputs)

    def validation_epoch_end(self, outputs) -> None:
        return self.adapter.validation_epoch_end(outputs)

    def test_epoch_end(self, outputs) -> None:
        return self.adapter.test_epoch_end(outputs)

    def process_train_dataset(
        self,
        dataset: InputBase,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = True,
        drop_last: bool = True,
        sampler: Optional[Sampler] = None,
        persistent_workers: bool = False,
        input_transform: Optional[InputTransform] = None,
        trainer: Optional["flash.Trainer"] = None,
    ) -> DataLoader:
        return self.adapter.process_train_dataset(
            dataset,
            batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
            persistent_workers=persistent_workers,
            input_transform=input_transform,
            trainer=trainer,
        )

    def process_val_dataset(
        self,
        dataset: InputBase,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = False,
        drop_last: bool = False,
        sampler: Optional[Sampler] = None,
        persistent_workers: bool = False,
        input_transform: Optional[InputTransform] = None,
        trainer: Optional["flash.Trainer"] = None,
    ) -> DataLoader:
        return self.adapter.process_val_dataset(
            dataset,
            batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
            persistent_workers=persistent_workers,
            input_transform=input_transform,
            trainer=trainer,
        )

    def process_test_dataset(
        self,
        dataset: InputBase,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = False,
        drop_last: bool = False,
        sampler: Optional[Sampler] = None,
        persistent_workers: bool = False,
        input_transform: Optional[InputTransform] = None,
        trainer: Optional["flash.Trainer"] = None,
    ) -> DataLoader:
        return self.adapter.process_test_dataset(
            dataset,
            batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
            persistent_workers=persistent_workers,
            input_transform=input_transform,
            trainer=trainer,
        )

    def process_predict_dataset(
        self,
        dataset: InputBase,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = False,
        drop_last: bool = False,
        sampler: Optional[Sampler] = None,
        persistent_workers: bool = False,
        input_transform: Optional[InputTransform] = None,
        trainer: Optional["flash.Trainer"] = None,
    ) -> DataLoader:
        return self.adapter.process_predict_dataset(
            dataset,
            batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
            persistent_workers=persistent_workers,
            input_transform=input_transform,
            trainer=trainer,
        )
