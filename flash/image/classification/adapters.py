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
import os
from collections import defaultdict
from functools import partial
from typing import Any, Callable, List, Optional, Type

import torch
from lightning_utilities.core.rank_zero import WarningCache
from pytorch_lightning import LightningModule
from pytorch_lightning.trainer.states import TrainerFn
from torch import nn, Tensor
from torch.utils.data import DataLoader, IterableDataset, Sampler

import flash
from flash.core.adapter import Adapter, AdapterTask
from flash.core.data.io.input import DataKeys, InputBase
from flash.core.data.io.input_transform import InputTransform
from flash.core.model import Task
from flash.core.registry import FlashRegistry
from flash.core.utilities.compatibility import accelerator_connector
from flash.core.utilities.imports import _LEARN2LEARN_AVAILABLE, _PL_GREATER_EQUAL_1_6_0
from flash.core.utilities.providers import _LEARN2LEARN
from flash.core.utilities.stability import beta
from flash.core.utilities.url_error import catch_url_error
from flash.image.classification.integrations.learn2learn import TaskDataParallel, TaskDistributedDataParallel

if _PL_GREATER_EQUAL_1_6_0:
    from pytorch_lightning.strategies import DataParallelStrategy, DDPSpawnStrategy, DDPStrategy
else:
    from pytorch_lightning.plugins import DataParallelPlugin, DDPPlugin, DDPSpawnPlugin

warning_cache = WarningCache()


if _LEARN2LEARN_AVAILABLE:
    import learn2learn as l2l
    from learn2learn.data.transforms import RemapLabels as Learn2LearnRemapLabels
else:

    class Learn2LearnRemapLabels:
        pass


class RemapLabels(Learn2LearnRemapLabels):
    def remap(self, data, mapping):
        # remap needs to be adapted to Flash API.
        data[DataKeys.TARGET] = mapping(data[DataKeys.TARGET])
        return data


class Model(nn.Module):
    def __init__(self, backbone: nn.Module, head: Optional[nn.Module]):
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


@beta("The Learn2Learn integration is currently in Beta.")
class Learn2LearnAdapter(Adapter):

    required_extras: str = "image"

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        algorithm_cls: Type[LightningModule],
        ways: int,
        shots: int,
        meta_batch_size: int,
        queries: int = 1,
        num_task: int = -1,
        epoch_length: Optional[int] = None,
        test_epoch_length: Optional[int] = None,
        test_ways: Optional[int] = None,
        test_shots: Optional[int] = None,
        test_queries: Optional[int] = None,
        test_num_task: Optional[int] = None,
        default_transforms_fn: Optional[Callable] = None,
        seed: int = 42,
        **algorithm_kwargs,
    ):
        """The ``Learn2LearnAdapter`` is an :class:`~flash.core.adapter.Adapter` for integrating with `learn 2
        learn` library (https://github.com/learnables/learn2learn).

        Args:
            task: Task to be used. This adapter should work with any Flash Classification task
            backbone: Feature extractor to be used.
            head: Predictive head.
            algorithm_cls: Algorithm class coming
                from: https://github.com/learnables/learn2learn/tree/master/learn2learn/algorithms/lightning
            ways: Number of classes conserved for generating the task.
            shots: Number of samples used for adaptation.
            meta_batch_size: Number of task to be sampled and optimized over before doing a meta optimizer step.
            queries: Number of samples used for computing the meta loss after the adaption on the `shots` samples.
            num_task: Total number of tasks to be sampled during training. If -1, a new task will always be sampled.
            epoch_length: Total number of tasks to be sampled to make an epoch.
            test_ways: Number of classes conserved for generating the validation and testing task.
            test_shots: Number of samples used for adaptation during validation and testing phase.
            test_queries: Number of samples used for computing the meta loss during validation or testing
                after the adaption on `shots` samples.
            epoch_length: Total number of tasks to be sampled to make an epoch during validation and testing phase.
            default_transforms_fn: A Callable to create the task transform.
                The callable should take the dataset, ways and shots as arguments.
            algorithm_kwargs: Keyword arguments to be provided to the algorithm class from learn2learn
        """

        super().__init__()

        self.backbone = backbone
        self.head = head
        self.algorithm_cls = algorithm_cls
        self.meta_batch_size = meta_batch_size

        self.num_task = num_task
        self.default_transforms_fn = default_transforms_fn
        self.seed = seed
        self.epoch_length = epoch_length or meta_batch_size

        self.ways = ways
        self.shots = shots
        self.queries = queries

        self.test_ways = test_ways or ways
        self.test_shots = test_shots or shots
        self.test_queries = test_queries or queries
        self.test_num_task = test_num_task or num_task
        self.test_epoch_length = test_epoch_length or self.epoch_length

        params = inspect.signature(self.algorithm_cls).parameters

        algorithm_kwargs["train_ways"] = ways
        algorithm_kwargs["train_shots"] = shots
        algorithm_kwargs["train_queries"] = queries

        algorithm_kwargs["test_ways"] = self.test_ways
        algorithm_kwargs["test_shots"] = self.test_shots
        algorithm_kwargs["test_queries"] = self.test_queries

        if "model" in params:
            algorithm_kwargs["model"] = Model(backbone=backbone, head=head)

        if "features" in params:
            algorithm_kwargs["features"] = Model(backbone=backbone, head=None)

        if "classifier" in params:
            algorithm_kwargs["classifier"] = head

        self.model = self.algorithm_cls(**algorithm_kwargs)

        # Patch log to avoid error with learn2learn and PL 1.5
        self.model.log = functools.partial(self._patch_log, self.model.log)

        # this algorithm requires a special treatment
        self._algorithm_has_validated = self.algorithm_cls != l2l.algorithms.LightningPrototypicalNetworks

    def _patch_log(self, log, *args, on_step: Optional[bool] = None, on_epoch: Optional[bool] = None, **kwargs):
        if not on_step and not on_epoch:
            on_epoch = True
        return log(*args, on_step=on_step, on_epoch=on_epoch, **kwargs)

    def _default_transform(self, dataset, ways: int, shots: int, queries) -> List[Callable]:
        return [
            l2l.data.transforms.FusedNWaysKShots(dataset, n=ways, k=shots + queries),
            l2l.data.transforms.LoadData(dataset),
            RemapLabels(dataset),
            l2l.data.transforms.ConsecutiveLabels(dataset),
        ]

    @staticmethod
    def _labels_to_indices(data):
        out = defaultdict(list)
        for idx, sample in enumerate(data):
            label = sample[DataKeys.TARGET]
            if torch.is_tensor(label):
                label = label.item()
            out[label].append(idx)
        return out

    def _convert_dataset(
        self,
        trainer: "flash.Trainer",
        dataset: InputBase,
        ways: int,
        shots: int,
        queries: int,
        num_workers: int,
        num_task: int,
        epoch_length: int,
    ):
        if trainer is None:
            raise ValueError(
                "The Learn2Learn integration requires the `Trainer` to be passed to the `process_*_dataset` method."
            )

        if isinstance(dataset, InputBase):

            metadata = getattr(dataset, "data", None)
            if metadata is None or (metadata is not None and not isinstance(dataset.data, list)):
                raise TypeError("Only dataset built out of metadata is supported.")

            labels_to_indices = self._labels_to_indices(dataset.data)

            if len(labels_to_indices) < ways:
                raise ValueError("Provided `ways` should be lower or equal to number of classes within your dataset.")

            if min(len(indice) for indice in labels_to_indices.values()) < (shots + queries):
                raise ValueError(
                    "Provided `shots + queries` should be lower than the lowest number of sample per class."
                )

            # convert the dataset to MetaDataset
            dataset = l2l.data.MetaDataset(dataset, indices_to_labels=None, labels_to_indices=labels_to_indices)

            transform_fn = self.default_transforms_fn or self._default_transform

            taskset = l2l.data.TaskDataset(
                dataset=dataset,
                task_transforms=transform_fn(dataset, ways=ways, shots=shots, queries=queries),
                num_tasks=num_task,
                task_collate=self._identity_task_collate_fn,
            )

        if _PL_GREATER_EQUAL_1_6_0:
            is_ddp_or_ddp_spawn = isinstance(
                trainer.strategy,
                (DDPStrategy, DDPSpawnStrategy),
            )
        else:
            is_ddp_or_ddp_spawn = isinstance(
                trainer.training_type_plugin,
                (DDPPlugin, DDPSpawnPlugin),
            )
        if is_ddp_or_ddp_spawn:
            # when running in a distributed data parallel way,
            # we are actually sampling one task per device.
            dataset = TaskDistributedDataParallel(
                taskset=taskset,
                global_rank=trainer.global_rank,
                world_size=trainer.world_size,
                num_workers=num_workers,
                epoch_length=epoch_length,
                seed=os.getenv("PL_GLOBAL_SEED", self.seed),
                requires_divisible=trainer.training,
            )
            self.trainer.accumulated_grad_batches = self.meta_batch_size / trainer.world_size
        else:
            devices = 1
            if _PL_GREATER_EQUAL_1_6_0:
                is_data_parallel = isinstance(trainer.strategy, DataParallelStrategy)
            else:
                is_data_parallel = isinstance(trainer.training_type_plugin, DataParallelPlugin)
            if is_data_parallel:
                # when using DP, we need to sample n tasks, so it can split across multiple devices.
                devices = accelerator_connector(trainer).devices
            dataset = TaskDataParallel(taskset, epoch_length=epoch_length, devices=devices, collate_fn=None)
            self.trainer.accumulated_grad_batches = self.meta_batch_size / devices

        return dataset

    @staticmethod
    def _identity_task_collate_fn(x: Any) -> Any:
        return x

    @classmethod
    @catch_url_error
    def from_task(
        cls,
        *args,
        task: AdapterTask,
        backbone: nn.Module,
        head: nn.Module,
        algorithm: Type[LightningModule],
        **kwargs,
    ) -> Adapter:
        if "meta_batch_size" not in kwargs:
            raise TypeError(
                "The `meta_batch_size` should be provided as training_strategy_kwargs={'meta_batch_size'=...}. "
                "This is equivalent to the epoch length."
            )
        if "shots" not in kwargs:
            raise TypeError(
                "The `shots` should be provided training_strategy_kwargs={'shots'=...}. "
                "This is equivalent to the number of sample per label to select within a task."
            )
        adapter = cls(backbone, head, algorithm, **kwargs)
        adapter.__dict__["_task"] = task
        return adapter

    def training_step(self, batch, batch_idx) -> Any:
        input = (batch[DataKeys.INPUT], batch[DataKeys.TARGET])
        return self.model.training_step(input, batch_idx)

    def validation_step(self, batch, batch_idx):
        # Should be True only for trainer.validate
        if self.trainer.state.fn == TrainerFn.VALIDATING:
            self._algorithm_has_validated = True
        input = (batch[DataKeys.INPUT], batch[DataKeys.TARGET])
        return self.model.validation_step(input, batch_idx)

    def validation_epoch_end(self, outpus: Any):
        self.model.validation_epoch_end(outpus)

    def test_step(self, batch, batch_idx):
        input = (batch[DataKeys.INPUT], batch[DataKeys.TARGET])
        return self.model.test_step(input, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self.model.predict_step(batch[DataKeys.INPUT], batch_idx, dataloader_idx=dataloader_idx)

    def _sanetize_batch_size(self, batch_size: int) -> int:
        if batch_size != 1:
            warning_cache.warn(
                "When using a meta-learning training_strategy, the batch_size should be set to 1. "
                "HINT: You can modify the `meta_batch_size` to 100 for example by doing "
                f"{type(self._task)}" + "(training_strategies_kwargs={'meta_batch_size': 100})"
            )
        return 1

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
        dataset = self._convert_dataset(
            trainer=trainer,
            dataset=dataset,
            ways=self.ways,
            shots=self.shots,
            queries=self.queries,
            num_workers=num_workers,
            num_task=self.num_task,
            epoch_length=self.epoch_length,
        )
        if isinstance(dataset, IterableDataset):
            shuffle = False
            sampler = None
        return super().process_train_dataset(
            dataset,
            self._sanetize_batch_size(batch_size),
            num_workers=num_workers,
            pin_memory=False,
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
        dataset = self._convert_dataset(
            trainer=trainer,
            dataset=dataset,
            ways=self.test_ways,
            shots=self.test_shots,
            queries=self.test_queries,
            num_workers=num_workers,
            num_task=self.test_num_task,
            epoch_length=self.test_epoch_length,
        )
        if isinstance(dataset, IterableDataset):
            shuffle = False
            sampler = None
        return super().process_val_dataset(
            dataset,
            self._sanetize_batch_size(batch_size),
            num_workers=num_workers,
            pin_memory=False,
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
        dataset = self._convert_dataset(
            trainer=trainer,
            dataset=dataset,
            ways=self.test_ways,
            shots=self.test_shots,
            queries=self.test_queries,
            num_workers=num_workers,
            num_task=self.test_num_task,
            epoch_length=self.test_epoch_length,
        )
        if isinstance(dataset, IterableDataset):
            shuffle = False
            sampler = None
        return super().process_test_dataset(
            dataset,
            self._sanetize_batch_size(batch_size),
            num_workers=num_workers,
            pin_memory=False,
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

        if not self._algorithm_has_validated:
            raise RuntimeError(
                "This training strategy needs to be validated before it can be used for prediction."
                " Call trainer.validate(...)."
            )

        return super().process_predict_dataset(
            dataset,
            batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
            persistent_workers=persistent_workers,
            input_transform=input_transform,
        )


class DefaultAdapter(Adapter):
    """The ``DefaultAdapter`` is an :class:`~flash.core.adapter.Adapter`."""

    required_extras: str = "image"

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()

        self.backbone = backbone
        self.head = head

    @classmethod
    @catch_url_error
    def from_task(
        cls,
        *args,
        task: AdapterTask,
        backbone: nn.Module,
        head: nn.Module,
        **kwargs,
    ) -> Adapter:
        adapter = cls(backbone, head)
        adapter.__dict__["_task"] = task
        return adapter

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DataKeys.INPUT], batch[DataKeys.TARGET])
        return Task.training_step(self._task, batch, batch_idx)

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DataKeys.INPUT], batch[DataKeys.TARGET])
        return Task.validation_step(self._task, batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DataKeys.INPUT], batch[DataKeys.TARGET])
        return Task.test_step(self._task, batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        batch[DataKeys.PREDS] = Task.predict_step(
            self._task, batch[DataKeys.INPUT], batch_idx, dataloader_idx=dataloader_idx
        )
        return batch

    def forward(self, x) -> Tensor:
        # TODO: Resolve this hack
        if x.dim() == 3:
            x = x.unsqueeze(0)
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
