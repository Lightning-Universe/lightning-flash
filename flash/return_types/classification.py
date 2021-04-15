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
from abc import ABC
from typing import Any, List, Union

import torch

from flash.return_types.base import ReturnType


class ClassificationReturnType(ReturnType, ABC):
    """"""


class Logits(ClassificationReturnType):

    def convert(self, sample: Any) -> Any:
        return sample


class Probabilities(ClassificationReturnType):

    def convert(self, sample: Any) -> Any:
        return torch.softmax(sample, -1)


class Classes(ClassificationReturnType):

    def convert(self, sample: Any) -> Union[int, List[int]]:
        return torch.argmax(sample, -1).tolist()


class Labels(Classes):

    def __init__(self, labels: List[str]):
        super().__init__()
        self._labels = labels

    def convert(self, sample: Any) -> Union[str, List[str]]:
        argmax = super().convert(sample)
        if isinstance(argmax, List):
            return [self._labels[i] for i in argmax]
        return self._labels[argmax]
