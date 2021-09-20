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
import inspect
import os
from collections import defaultdict
from functools import partial
from typing import Any, Callable, List, Optional, Type

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.plugins import DataParallelPlugin, DDPPlugin, DDPSpawnPlugin
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.warnings import WarningCache
from torch.utils.data import DataLoader, IterableDataset, Sampler

import flash
from flash.core.adapter import Adapter, AdapterTask
from flash.core.data.auto_dataset import BaseAutoDataset
from flash.core.data.data_source import DefaultDataKeys
from flash.core.model import Task
from flash.core.registry import FlashRegistry
from flash.core.utilities.url_error import catch_url_error

warning_cache = WarningCache()


class NoModule:
    """This class is used to prevent nn.Module infinite recursion."""

    def __init__(self, task):
        self.task = task

    def __getattr__(self, key):
        if key != "task":
            return getattr(self.task, key)
        return self.task

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "task":
            object.__setattr__(self, key, value)
            return
        setattr(self.task, key, value)


class Model(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, head: Optional[torch.nn.Module]):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        if x.dim() == 4:
            x = x.mean(-1).mean(-1)
        if self.head is None:
            return x
        return self.head(x)


class DefaultAdapter(Adapter):
    """The ``DefaultAdapter`` is an :class:`~flash.core.adapter.Adapter`."""

    required_extras: str = "text"

    def __init__(self, task: AdapterTask, backbone: torch.nn.Module, head: torch.nn.Module):
        super().__init__()

        self._task = NoModule(task)
        self.backbone = backbone
        self.head = head

    @classmethod
    @catch_url_error
    def from_task(
        cls,
        *args,
        task: AdapterTask,
        backbone: torch.nn.Module,
        head: torch.nn.Module,
        **kwargs,
    ) -> Adapter:
        return cls(task, backbone, head)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return Task.training_step(self._task.task, batch, batch_idx)

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return Task.validation_step(self._task.task, batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return Task.test_step(self._task.task, batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        batch[DefaultDataKeys.PREDS] = Task.predict_step(
            self._task.task, (batch[DefaultDataKeys.INPUT]), batch_idx, dataloader_idx=dataloader_idx
        )
        return batch

    def forward(self, x) -> torch.Tensor:
        x = self.backbone(x)
        return self.head(x)


TRAINING_STRATEGIES = FlashRegistry("training_strategies")
TRAINING_STRATEGIES(name="default", fn=partial(DefaultAdapter.from_task))
