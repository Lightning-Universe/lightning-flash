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
from inspect import signature
from typing import Any, Callable, Generic, Iterable, Optional, Sequence, TypeVar

from pytorch_lightning.trainer.states import RunningStage
from torch.utils.data import Dataset, IterableDataset

import flash
from flash.core.data.utils import CurrentRunningStageFuncContext

DATA_TYPE = TypeVar("DATA_TYPE")


class BaseAutoDataset(Generic[DATA_TYPE]):
    """The ``BaseAutoDataset`` class wraps the output of a call to :meth:`~flash.core.data.data_source.DataSource.load_data`
    and a :class:`~fash.data.data_source.DataSource` and provides the ``_call_load_sample`` method to call
    :meth:`~flash.core.data.data_source.DataSource.load_sample` with the correct
    :class:`~flash.core.data.utils.CurrentRunningStageFuncContext` for the current ``running_stage``.
    Inheriting classes are responsible for extracting samples from ``data`` to be given to ``_call_load_sample``.

    Args:
        data: The output of a call to :meth:`~flash.core.data.data_source.DataSource.load_data`.
        data_source: The :class:`~flash.core.data.data_source.DataSource` which has the ``load_sample`` method.
        running_stage: The current running stage.
    """

    DATASET_KEY = "dataset"

    def __init__(
        self,
        data: DATA_TYPE,
        data_source: "flash.core.data.data_source.DataSource",
        running_stage: RunningStage,
    ) -> None:
        super().__init__()

        self.data = data
        self.data_source = data_source

        self._running_stage = None
        self.running_stage = running_stage

    @property
    def running_stage(self) -> RunningStage:
        return self._running_stage

    @running_stage.setter
    def running_stage(self, running_stage: RunningStage) -> None:
        from flash.core.data.data_pipeline import DataPipeline  # noqa F811
        from flash.core.data.data_source import DataSource  # noqa F811 # TODO: something better than this

        self._running_stage = running_stage

        self._load_sample_context = CurrentRunningStageFuncContext(self.running_stage, "load_sample", self.data_source)

        self.load_sample: Callable[[DATA_TYPE, Optional[Any]], Any] = getattr(
            self.data_source,
            DataPipeline._resolve_function_hierarchy(
                "load_sample",
                self.data_source,
                self.running_stage,
                DataSource,
            ),
        )

    def _call_load_sample(self, sample: Any) -> Any:
        if self.load_sample:
            if isinstance(sample, dict):
                sample = dict(**sample)
            with self._load_sample_context:
                parameters = signature(self.load_sample).parameters
                if len(parameters) > 1 and self.DATASET_KEY in parameters:
                    sample = self.load_sample(sample, self)
                else:
                    sample = self.load_sample(sample)
        return sample


class AutoDataset(BaseAutoDataset[Sequence], Dataset):
    """The ``AutoDataset`` is a ``BaseAutoDataset`` and a :class:`~torch.utils.data.Dataset`.

    The `data` argument must be a ``Sequence`` (it must have a length).
    """

    def __getitem__(self, index: int) -> Any:
        return self._call_load_sample(self.data[index])

    def __len__(self) -> int:
        return len(self.data)


class IterableAutoDataset(BaseAutoDataset[Iterable], IterableDataset):
    """The ``IterableAutoDataset`` is a ``BaseAutoDataset`` and a :class:`~torch.utils.data.IterableDataset`.

    The `data` argument must be an ``Iterable``.
    """

    def __iter__(self):
        self.data_iter = iter(self.data)
        return self

    def __next__(self) -> Any:
        return self._call_load_sample(next(self.data_iter))
