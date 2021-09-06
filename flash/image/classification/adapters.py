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
import random
from collections import defaultdict
from typing import Any, Callable, Optional, Type

import torch
from pytorch_lightning import LightningModule
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


class UserTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, task_description):
        for dd in task_description:
            dd.transforms.append(self.transforms)
        return task_description


class Model(torch.nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        if x.dim() == 4:
            x = x.mean(-1).mean(-1)
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
        algorithm: Type[LightningModule],
        train_samples: int,
        train_ways: int,
        test_samples: int,
        test_ways: int,
        **algorithm_kwargs,
    ):
        super().__init__()

        self._task = NoModule(task)
        self.backbone = backbone
        self.head = head
        self.algorithm = algorithm
        self.train_samples = train_samples
        self.train_ways = train_ways
        self.test_samples = test_samples
        self.test_ways = test_ways

        self.model = self.algorithm(Model(backbone=backbone, head=head), **algorithm_kwargs)

    def _train_transforms(self, dataset):
        return [
            l2l.data.transforms.FusedNWaysKShots(dataset, n=self.train_ways, k=self.train_samples),
            l2l.data.transforms.LoadData(dataset),
            RemapLabels(dataset),
            # l2l.data.transforms.ConsecutiveLabels(dataset),
            # l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
        ]

    def _evaluation_transforms(self, dataset):
        return [
            l2l.data.transforms.FusedNWaysKShots(dataset, n=self.test_ways, k=self.test_samples),
            l2l.data.transforms.LoadData(dataset),
            l2l.data.transforms.RemapLabels(dataset),
            l2l.data.transforms.ConsecutiveLabels(dataset),
            l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0]),
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

        # convert the dataset to MetaDataset
        dataset = l2l.data.MetaDataset(
            dataset, indices_to_labels=indices_to_labels, labels_to_indices=labels_to_indices
        )
        taskset = l2l.data.TaskDataset(
            dataset=dataset,
            task_transforms=self._train_transforms(dataset),
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
        input = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return self.model.training_step(input, batch_idx)

    def test_step(self, batch, batch_idx):
        input = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return self.model.training_step(input, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        raise NotImplementedError

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
            self.convert_dataset(dataset, collate_fn),
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
            self.convert_dataset(dataset, collate_fn),
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
        return super().process_predict_dataset(
            self.convert_dataset(dataset, collate_fn),
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


def _fn():
    pass


TRAINING_STRATEGIES = FlashRegistry("training_strategies")
TRAINING_STRATEGIES(name="default", fn=_fn, adapter=DefaultAdapter, algorithm=str)

if _LEARN2LEARN_AVAILABLE:
    from learn2learn import algorithms

    for algorithm in dir(algorithms):
        try:
            if "lightning" in algorithm.lower() and issubclass(getattr(algorithms, algorithm), LightningModule):
                TRAINING_STRATEGIES(
                    name=algorithm.lower().replace("lightning", ""),
                    fn=_fn,
                    adapter=Learn2LearnAdapter,
                    algorithm=getattr(algorithms, algorithm),
                    providers=[_LEARN2LEARN],
                )
        except Exception:
            pass
