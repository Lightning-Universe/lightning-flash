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
import functools
import inspect
import random
from collections import defaultdict
from functools import partial
from typing import Any, Callable, List, Optional, Type

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader, Sampler

from flash.core.adapter import Adapter, AdapterTask
from flash.core.data.auto_dataset import BaseAutoDataset
from flash.core.data.data_source import DefaultDataKeys
from flash.core.model import Task
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _LEARN2LEARN_AVAILABLE
from flash.core.utilities.providers import _LEARN2LEARN
from flash.core.utilities.url_error import catch_url_error

if _LEARN2LEARN_AVAILABLE:
    import learn2learn as l2l

    class RemapLabels(l2l.data.transforms.TaskTransform):
        def __init__(self, dataset, shuffle=True):
            super().__init__(dataset)
            self.dataset = dataset
            self.shuffle = shuffle

        def remap(self, data, mapping):
            data[DefaultDataKeys.TARGET] = mapping(data[DefaultDataKeys.TARGET])
            return data

        def __call__(self, task_description):
            if task_description is None:
                task_description = self.new_task()
            labels = list({self.dataset.indices_to_labels[dd.index] for dd in task_description})
            if self.shuffle:
                random.shuffle(labels)

            def mapping(x):
                return labels.index(x)

            for dd in task_description:
                remap = functools.partial(self.remap, mapping=mapping)
                dd.transforms.append(remap)
            return task_description


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


class Epochifier:
    def __init__(self, tasks, length):
        self.tasks = tasks
        self.length = length

    def __getitem__(self, *args, **kwargs):
        return self.tasks.sample()

    def __len__(self):
        return self.length


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


class Learn2LearnAdapter(Adapter):
    """The ``Learn2LearnAdapter`` is an :class:`~flash.core.adapter.Adapter` for integrating with learn 2 learn
    library."""

    required_extras: str = "image"

    def __init__(
        self,
        task: AdapterTask,
        backbone: torch.nn.Module,
        head: torch.nn.Module,
        algorithm_cls: Type[LightningModule],
        ways: int,
        kshots: int,
        queries: int = 1,
        **algorithm_kwargs,
    ):
        super().__init__()

        self._task = NoModule(task)
        self.backbone = backbone
        self.head = head
        self.algorithm_cls = algorithm_cls
        self.ways = ways
        self.kshots = kshots
        self.queries = queries

        params = inspect.signature(self.algorithm_cls).parameters

        algorithm_kwargs["train_ways"] = ways
        algorithm_kwargs["test_ways"] = ways

        algorithm_kwargs["train_shots"] = kshots - queries
        algorithm_kwargs["test_shots"] = kshots - queries

        algorithm_kwargs["train_queries"] = queries
        algorithm_kwargs["test_queries"] = queries

        if "model" in params:
            algorithm_kwargs["model"] = Model(backbone=backbone, head=head)

        if "features" in params:
            algorithm_kwargs["features"] = Model(backbone=backbone, head=None)

        if "classifier" in params:
            algorithm_kwargs["classifier"] = head

        self.model = self.algorithm_cls(**algorithm_kwargs)

        # this algorithm requires a special treatment
        self._algorithm_has_validated = self.algorithm_cls != l2l.algorithms.LightningPrototypicalNetworks

    def _default_transform(self, dataset) -> List[Callable]:
        return [
            l2l.data.transforms.FusedNWaysKShots(dataset, n=self.ways, k=self.kshots),
            l2l.data.transforms.LoadData(dataset),
            RemapLabels(dataset),
            l2l.data.transforms.ConsecutiveLabels(dataset),
        ]

    @property
    def task(self) -> Task:
        return self._task.task

    def convert_dataset(self, dataset):
        metadata = getattr(dataset, "data", None)
        if metadata is None or (metadata is not None and not isinstance(dataset.data, list)):
            raise MisconfigurationException("Only dataset built out of metadata is supported.")

        indices_to_labels = {index: sample[DefaultDataKeys.TARGET] for index, sample in enumerate(dataset.data)}
        labels_to_indices = defaultdict(list)
        for idx, label in indices_to_labels.items():
            labels_to_indices[label].append(idx)

        if len(labels_to_indices) < self.ways:
            raise MisconfigurationException(
                "Provided `ways` should be lower or equal to number of classes within your dataset."
            )

        if min(len(indice) for indice in labels_to_indices.values()) < (self.kshots + self.queries):
            raise MisconfigurationException(
                "Provided `kshots` should be lower than the lowest number of sample per class."
            )

        # convert the dataset to MetaDataset
        dataset = l2l.data.MetaDataset(
            dataset, indices_to_labels=indices_to_labels, labels_to_indices=labels_to_indices
        )
        taskset = l2l.data.TaskDataset(
            dataset=dataset,
            task_transforms=self._default_transform(dataset),
            num_tasks=-1,
            task_collate=self._identity_fn,
        )
        dataset = Epochifier(taskset, 100)
        return dataset

    @staticmethod
    def _identity_fn(x: Any) -> Any:
        return x

    @classmethod
    @catch_url_error
    def from_task(
        cls,
        *args,
        task: AdapterTask,
        backbone: torch.nn.Module,
        head: torch.nn.Module,
        algorithm: Type[LightningModule],
        **kwargs,
    ) -> Adapter:
        return cls(task, backbone, head, algorithm, **kwargs)

    def training_step(self, batch, batch_idx) -> Any:
        input = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return self.model.training_step(input, batch_idx)

    def validation_step(self, batch, batch_idx):
        # Should be True only for trainer.validate
        if self.trainer.state.fn == TrainerFn.VALIDATING:
            self._algorithm_has_validated = True
        input = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return self.model.validation_step(input, batch_idx)

    def validation_epoch_end(self, outpus: Any):
        self.model.validation_epoch_end(outpus)

    def test_step(self, batch, batch_idx):
        input = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return self.model.test_step(input, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self.model.predict_step(batch[DefaultDataKeys.INPUT], batch_idx, dataloader_idx=dataloader_idx)

    def process_train_dataset(
        self,
        dataset: BaseAutoDataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        collate_fn: Callable,
        shuffle: bool,
        drop_last: bool,
        sampler: Optional[Sampler],
    ) -> DataLoader:
        assert batch_size == 1
        return super().process_train_dataset(
            self.convert_dataset(dataset),
            batch_size,
            num_workers,
            pin_memory,
            collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
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
        assert batch_size == 1
        return super().process_val_dataset(
            self.convert_dataset(dataset),
            batch_size,
            num_workers,
            pin_memory,
            collate_fn,
            shuffle,
            drop_last,
            sampler,
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
        assert batch_size == 1
        return super().process_test_dataset(
            self.convert_dataset(dataset),
            batch_size,
            num_workers,
            pin_memory,
            collate_fn,
            shuffle,
            drop_last,
            sampler,
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
        assert batch_size == 1

        if not self._algorithm_has_validated:
            raise MisconfigurationException(
                "This training_strategies requires to be validated. Call trainer.validate(...)."
            )

        return super().process_predict_dataset(
            dataset,
            batch_size,
            num_workers,
            pin_memory,
            collate_fn,
            shuffle,
            drop_last,
            sampler,
        )


class DefaultAdapter(Adapter):
    """The ``DefaultAdapter`` is an :class:`~flash.core.adapter.Adapter`."""

    required_extras: str = "image"

    def __init__(self, task: AdapterTask, backbone: torch.nn.Module, head: torch.nn.Module):
        super().__init__()

        self._task = NoModule(task)
        self.backbone = backbone
        self.head = head

    @property
    def task(self) -> Task:
        return self._task.task

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
        return Task.training_step(self.task, batch, batch_idx)

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return Task.validation_step(self.task, batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return Task.test_step(self.task, batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        batch[DefaultDataKeys.PREDS] = Task.predict_step(
            self.task, (batch[DefaultDataKeys.INPUT]), batch_idx, dataloader_idx=dataloader_idx
        )
        return batch

    def forward(self, x) -> torch.Tensor:
        x = self.backbone(x)
        if x.dim() == 4:
            x = x.mean(-1).mean(-1)
        return self.head(x)


TRAINING_STRATEGIES = FlashRegistry("training_strategies")
TRAINING_STRATEGIES(name="default", fn=partial(DefaultAdapter.from_task))

if _LEARN2LEARN_AVAILABLE:
    from learn2learn import algorithms

    for algorithm in dir(algorithms):
        # skip base class
        if algorithm == "LightningEpisodicModule":
            continue
        try:
            if "lightning" in algorithm.lower() and issubclass(getattr(algorithms, algorithm), LightningModule):
                TRAINING_STRATEGIES(
                    name=algorithm.lower().replace("lightning", ""),
                    fn=partial(Learn2LearnAdapter.from_task, algorithm=getattr(algorithms, algorithm)),
                    providers=[_LEARN2LEARN],
                )
        except Exception:
            pass
