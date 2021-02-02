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
from typing import Any, Union

import torch

from flash.core.data import TaskDataPipeline
from flash.core.model import Task


class ClassificationDataPipeline(TaskDataPipeline):

    def before_uncollate(self, batch: Union[torch.Tensor, tuple]) -> torch.Tensor:
        if isinstance(batch, tuple):
            batch = batch[0]
        return torch.softmax(batch, -1)

    def after_uncollate(self, samples: Any) -> Any:
        return torch.argmax(samples, -1).tolist()


class ClassificationTask(Task):

    @staticmethod
    def default_pipeline() -> ClassificationDataPipeline:
        return ClassificationDataPipeline()
