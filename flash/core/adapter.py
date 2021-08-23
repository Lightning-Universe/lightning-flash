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

from torch import nn
from torch.utils.data import DataLoader, Sampler

import flash
from flash.core.data.auto_dataset import BaseAutoDataset
from flash.core.model import DatasetProcessor, ModuleWrapperBase, Task


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

    @property
    def backbone(self) -> nn.Module:
        return self.adapter.backbone

    def forward(self, x: Any) -> Any:
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
        dataset: BaseAutoDataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        collate_fn: Callable,
        shuffle: bool = False,
        drop_last: bool = True,
        sampler: Optional[Sampler] = None,
    ) -> DataLoader:
        return self.adapter.process_train_dataset(
            dataset, batch_size, num_workers, pin_memory, collate_fn, shuffle, drop_last, sampler
        )

    def process_val_dataset(
        self,
        dataset: BaseAutoDataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        collate_fn: Callable,
        shuffle: bool = False,
        drop_last: bool = False,
        sampler: Optional[Sampler] = None,
    ) -> DataLoader:
        return self.adapter.process_val_dataset(
            dataset, batch_size, num_workers, pin_memory, collate_fn, shuffle, drop_last, sampler
        )

    def process_test_dataset(
        self,
        dataset: BaseAutoDataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        collate_fn: Callable,
        shuffle: bool = False,
        drop_last: bool = False,
        sampler: Optional[Sampler] = None,
    ) -> DataLoader:
        return self.adapter.process_test_dataset(
            dataset, batch_size, num_workers, pin_memory, collate_fn, shuffle, drop_last, sampler
        )

    def process_predict_dataset(
        self,
        dataset: BaseAutoDataset,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        collate_fn: Callable = lambda x: x,
        shuffle: bool = False,
        drop_last: bool = True,
        sampler: Optional[Sampler] = None,
    ) -> DataLoader:
        return self.adapter.process_predict_dataset(
            dataset, batch_size, num_workers, pin_memory, collate_fn, shuffle, drop_last, sampler
        )
