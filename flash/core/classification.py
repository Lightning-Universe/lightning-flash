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
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Union

import torch
import torch.nn.functional as F

from flash.core.model import Task
from flash.data.process import Serializer, ProcessState


@dataclass(unsafe_hash=True, frozen=True)
class ClassificationState(ProcessState):
    classes: List[str]


class ClassificationTask(Task):
    def __init__(
            self,
            *args,
            serializer: Optional[Union[Serializer, Mapping[str, Serializer]]] = None,
            **kwargs,
    ) -> None:
        super().__init__(*args, serializer=serializer or Classes(), **kwargs)

    def to_metrics_format(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, -1)


class Logits(Serializer):

    def serialize(self, sample: Any) -> Any:
        return sample.tolist()


class Probabilities(Serializer):

    def serialize(self, sample: Any) -> Any:
        return torch.softmax(sample, -1).tolist()


class Classes(Serializer):

    def serialize(self, sample: Any) -> Union[int, List[int]]:
        return torch.argmax(sample, -1).tolist()


class Labels(Classes):

    def __init__(self, labels: Optional[List[str]] = None):
        super().__init__()
        self._labels = labels

    def serialize(self, sample: Any) -> Union[str, List[str]]:
        if self._labels is not None:
            labels = self._labels
        else:
            state = self.get_state(ClassificationState)
            if state is not None:
                labels = state.classes
            else:
                raise ValueError  # TODO: Better error

        argmax = super().serialize(sample)
        if isinstance(argmax, List):
            return [labels[i] for i in argmax]
        return labels[argmax]
