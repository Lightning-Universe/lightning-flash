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

"""
Note: This file will be deleted once
https://github.com/learnables/learn2learn/pull/257/files is merged within Learn2Learn.
"""

from typing import Any, Callable, Optional

import pytorch_lightning as pl
from torch.utils.data import IterableDataset
from torch.utils.data._utils.worker import get_worker_info

from flash.core.data.utilities.collate import default_collate
from flash.core.utilities.imports import requires


class TaskDataParallel(IterableDataset):
    @requires("learn2learn")
    def __init__(
        self,
        tasks: Any,
        epoch_length: int,
        devices: int = 1,
        collate_fn: Optional[Callable] = default_collate,
    ):
        """This class is used to sample epoch_length tasks to represent an epoch.

        It should be used when using DataParallel

        Args:
            taskset: Dataset used to sample task.
            epoch_length: The expected epoch length. This requires to be divisible by devices.
            devices: Number of devices being used.
            collate_fn: The collate_fn to be applied on multiple tasks
        """
        self.tasks = tasks
        self.epoch_length = epoch_length
        self.devices = devices

        if epoch_length % devices != 0:
            raise Exception("The `epoch_length` should be the number of `devices`.")

        self.collate_fn = collate_fn
        self.counter = 0

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter >= len(self):
            raise StopIteration
        self.counter += self.devices
        tasks = []
        for _ in range(self.devices):
            for item in self.tasks.sample():
                tasks.append(item)
        if self.collate_fn:
            tasks = self.collate_fn(tasks)
        return tasks

    def __len__(self):
        return self.epoch_length


class TaskDistributedDataParallel(IterableDataset):
    @requires("learn2learn")
    def __init__(
        self,
        taskset: Any,
        global_rank: int,
        world_size: int,
        num_workers: int,
        epoch_length: int,
        seed: int,
        requires_divisible: bool = True,
    ):
        """This class is used to sample tasks in a distributed setting such as DDP with multiple workers.

        This won't work as expected if `num_workers = 0` and several dataloaders
        are being iterated on at the same time.

        Args:
            taskset: Dataset used to sample task.
            global_rank: Rank of the current process.
            world_size: Total of number of processes.
            num_workers: Number of workers to be provided to the DataLoader.
            epoch_length: The expected epoch length. This requires to be divisible by (num_workers * world_size).
            seed: The seed will be used on __iter__ call and should be the same for all processes.
        """
        self.taskset = taskset
        self.global_rank = global_rank
        self.world_size = world_size
        self.num_workers = 1 if num_workers == 0 else num_workers
        self.worker_world_size = self.world_size * self.num_workers
        self.epoch_length = epoch_length
        self.seed = seed
        self.iteration = 0
        self.requires_divisible = requires_divisible
        self.counter = 0

        if requires_divisible and epoch_length % self.worker_world_size != 0:
            raise Exception("The `epoch_length` should be divisible by `world_size`.")

    def __len__(self) -> int:
        return self.epoch_length // self.world_size

    @property
    def worker_id(self) -> int:
        worker_info = get_worker_info()
        return worker_info.id if worker_info else 0

    @property
    def worker_rank(self) -> int:
        is_global_zero = self.global_rank == 0
        return self.global_rank + self.worker_id + int(not is_global_zero and self.num_workers > 1)

    def __iter__(self):
        self.iteration += 1
        self.counter = 0
        pl.seed_everything(self.seed + self.iteration)
        return self

    def __next__(self):
        if self.counter >= len(self):
            raise StopIteration
        task_descriptions = []
        for _ in range(self.worker_world_size):
            task_descriptions.append(self.taskset.sample_task_description())

        data = self.taskset.get_task(task_descriptions[self.worker_rank])
        self.counter += 1
        return data
