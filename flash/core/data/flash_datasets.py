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
from typing import Any, Callable, Generic, Iterable, Optional, Sequence, Type, TypeVar

from pytorch_lightning.trainer.states import RunningStage
from torch.utils.data import Dataset, IterableDataset

from flash.core.data.properties import Properties

DATA_TYPE = TypeVar("DATA_TYPE")


class BaseDataset(Generic[DATA_TYPE], Properties):
    DATASET_KEY = "dataset"

    def load_data(self, data: Any) -> Any:
        return data

    def load_sample(self, data: Any) -> Any:
        return data

    DATASET_KEY = "dataset"

    def __init__(
        self,
    ) -> None:

        """The ``BaseDataset`` class wraps the output of a call to :meth:`~flash.core.data.data_source.DataSource.load_data`
        and a :class:`~fash.data.data_source.DataSource` and provides the ``_call_load_sample`` method to call
        :meth:`~flash.core.data.data_source.DataSource.load_sample` with the correct
        :class:`~flash.core.data.utils.CurrentRunningStageFuncContext` for the current ``running_stage``.
        Inheriting classes are responsible for extracting samples from ``data`` to be given to ``_call_load_sample``.

        Args:
            data: The output of a call to :meth:`~flash.core.data.data_source.DataSource.load_data`.
            data_source: The :class:`~flash.core.data.data_source.DataSource` which has the ``load_sample`` method.
            running_stage: The current running stage.
        """

        super().__init__()

    def setup(self, data: Any, running_stage: RunningStage) -> None:
        self.running_stage = running_stage
        self.data = self._load_data(data)

    @property
    def running_stage(self) -> RunningStage:
        return self._running_stage

    @running_stage.setter
    def running_stage(self, running_stage: RunningStage) -> None:
        self._running_stage = running_stage
        self.resolve_functions()

    def _resolve_functions(self, func_name: str, cls: Type["BaseDataset"]) -> None:
        from flash.core.data.data_pipeline import DataPipeline  # noqa F811

        function: Callable[[DATA_TYPE, Optional[Any]], Any] = getattr(
            self,
            DataPipeline._resolve_function_hierarchy(
                func_name,
                self,
                self.running_stage,
                cls,
            ),
        )
        setattr(self, f"_{func_name}", function)

    def _call_load_sample(self, sample: Any) -> Any:
        if self.load_sample:
            if isinstance(sample, dict):
                sample = dict(**sample)
                sample = self._load_sample(sample)
        return sample

    @classmethod
    def from_data(cls, data: Any, running_stage: RunningStage, **kwargs: Any) -> "BaseDataset":
        flash_dataset = cls(**kwargs)
        flash_dataset.setup(data, running_stage)
        return flash_dataset

    def resolve_functions(self):
        raise NotImplementedError

    _load_data = load_data
    _load_sample = load_sample


class FlashDataset(BaseDataset[Sequence], Dataset):
    """The ``FlashDataset`` is a ``BaseDataset`` and a :class:`~torch.utils.data.Dataset`.

    The `data` argument must be a ``Sequence`` (it must have a length).
    """

    def resolve_functions(self):
        self._resolve_functions("load_data", FlashDataset)
        self._resolve_functions("load_sample", FlashDataset)

    def __getitem__(self, index: int) -> Any:
        return self._call_load_sample(self.data[index])

    def __len__(self) -> int:
        return len(self.data)


class FlashIterableDataset(BaseDataset[Iterable], IterableDataset):
    """The ``IterableAutoDataset`` is a ``BaseDataset`` and a :class:`~torch.utils.data.IterableDataset`.

    The `data` argument must be an ``Iterable``.
    """

    def resolve_functions(self):
        self._resolve_functions("load_data", FlashIterableDataset)
        self._resolve_functions("load_sample", FlashIterableDataset)

    def __iter__(self):
        self.data_iter = iter(self.data)
        return self

    def __next__(self) -> Any:
        return self._call_load_sample(next(self.data_iter))
