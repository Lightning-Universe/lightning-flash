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
from typing import Any, Callable, Iterable, Mapping, Optional, Type, Union

from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from flash.core.data.properties import Properties
from flash.core.utilities.imports import _TORCH_DATA_AVAILABLE

if _TORCH_DATA_AVAILABLE:
    from torch.utils.data import IterDataPipe


class FlashDataPipes(Properties, IterDataPipe):

    DATASET_KEY = "dataset"

    data_pipe: Optional["FlashDataPipes"] = None

    # This function is resolved dynamically once the stage was been set
    current_process_data: Optional[Callable] = None  #

    @abstractmethod
    def process_data(self, data: Any) -> Union[Iterable, Mapping]:
        """The `load_data` hook should return either a Mapping or an Iterable.

        Override to add your dataset logic creation logic.
        """

    def __init__(self, running_stage: RunningStage) -> None:
        super().__init__()
        self.running_stage = running_stage

    @property
    def running_stage(self) -> Optional[RunningStage]:
        return self._running_stage

    @running_stage.setter
    def running_stage(self, running_stage: RunningStage) -> None:
        self._running_stage = running_stage
        self._resolve_functions("process_data", FlashDataPipes)

    def _resolve_functions(self, func_name: str, cls: Type["FlashDataPipes"]) -> None:
        from flash.core.data.data_pipeline import DataPipeline  # noqa F811

        function: Callable[[Any, Optional[Any]], Any] = getattr(
            self,
            DataPipeline._resolve_function_hierarchy(
                func_name,
                self,
                self.running_stage,
                cls,
            ),
        )
        setattr(self, f"current_{func_name}", function)

    @classmethod
    def from_data(
        cls,
        *args,
        running_stage: Optional[RunningStage] = None,
        **kwargs: Any,
    ) -> "FlashDataPipes":
        if not running_stage:
            raise MisconfigurationException(
                "You should provide a running_stage to your dataset"
                " `from pytorch_lightning.trainer.states import RunningStage`."
            )
        flash_dataset = cls(*args, running_stage=running_stage, **kwargs)
        return flash_dataset

    @classmethod
    def from_train_data(
        cls,
        *load_data_args,
        **dataset_kwargs: Any,
    ) -> "FlashDataPipes":
        return cls.from_data(*load_data_args, running_stage=RunningStage.TRAINING, **dataset_kwargs)

    @classmethod
    def from_val_data(
        cls,
        *load_data_args,
        **dataset_kwargs: Any,
    ) -> "FlashDataPipes":
        return cls.from_data(*load_data_args, running_stage=RunningStage.VALIDATING, **dataset_kwargs)

    @classmethod
    def from_test_data(
        cls,
        *load_data_args,
        **dataset_kwargs: Any,
    ) -> "FlashDataPipes":
        return cls.from_data(*load_data_args, running_stage=RunningStage.TESTING, **dataset_kwargs)

    @classmethod
    def from_predict_data(
        cls,
        *load_data_args,
        **dataset_kwargs: Any,
    ) -> "FlashDataPipes":
        return cls.from_data(*load_data_args, running_stage=RunningStage.PREDICTING, **dataset_kwargs)

    def __len__(self):
        return len(self.data_pipe)

    def __iter__(self):
        if self.data_pipe:
            for data in self.data_pipe:
                yield self.current_process_data(data)
        else:
            for data in self.current_process_data():
                yield data
