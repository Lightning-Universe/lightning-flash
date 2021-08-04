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
from typing import Any, Callable, Optional

from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.utils.data import DataLoader, Sampler

import flash
from flash.core.data.auto_dataset import BaseAutoDataset


class Wrapper:

    def __init__(self):
        super().__init__()

        self._children = []

    def __setattr__(self, key, value):
        if isinstance(value, (LightningModule, Adapter)):
            self._children.append(key)
        patched_attributes = ["_current_fx_name", "_current_hook_fx_name", "_results"]
        if isinstance(value, Trainer) or key in patched_attributes:
            if hasattr(self, "_children"):
                for child in self._children:
                    setattr(getattr(self, child), key, value)
        super().__setattr__(key, value)


class Adapter(Wrapper, nn.Module):

    @classmethod
    def from_task(cls, task: 'flash.Task', **kwargs) -> 'Adapter':
        pass

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

    def _process_dataset(
        self,
        dataset: BaseAutoDataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        collate_fn: Callable,
        shuffle: bool = False,
        drop_last: bool = True,
        sampler: Optional[Sampler] = None
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_fn
        )

    def process_train_dataset(
        self,
        dataset: BaseAutoDataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        collate_fn: Callable,
        shuffle: bool = False,
        drop_last: bool = True,
        sampler: Optional[Sampler] = None
    ) -> DataLoader:
        return self._process_dataset(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler
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
        sampler: Optional[Sampler] = None
    ) -> DataLoader:
        return self._process_dataset(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler
        )

    def process_test_dataset(
        self,
        dataset: BaseAutoDataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        collate_fn: Callable,
        shuffle: bool = False,
        drop_last: bool = True,
        sampler: Optional[Sampler] = None
    ) -> DataLoader:
        return self._process_dataset(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler
        )

    def process_predict_dataset(
        self,
        dataset: BaseAutoDataset,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        collate_fn: Callable = None,
        shuffle: bool = False,
        drop_last: bool = True,
        sampler: Optional[Sampler] = None
    ) -> DataLoader:
        return self._process_dataset(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler
        )
