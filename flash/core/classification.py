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
from typing import Any, Callable, Mapping, Optional, Sequence, Type, Union

import pytorch_lightning as pl
import torch
from torch.nn.functional import binary_cross_entropy, cross_entropy

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

    def __init__(
        self,
        num_classes: int,
        model: Optional[torch.nn.Module] = None,
        loss_fn: Optional[Callable] = None,
        multilabel: bool = False,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.SGD,
        metrics: Union[pl.metrics.Metric, Mapping, Sequence, None] = None,
        learning_rate: float = 1e-3,
    ):

        super().__init__(
            model=model, loss_fn=loss_fn, optimizer=optimizer, metrics=metrics, learning_rate=learning_rate
        )

        self.num_classes = num_classes
        self.multilabel = multilabel

        if isinstance(self.loss_fn, Mapping) and not self.loss_fn:
            self.loss_fn = self.default_loss_fn

    def step(self, batch: Sequence[torch.Tensor], batch_idx: int) -> torch.Tensor:
        if self.multilabel:
            x, y = batch
            if isinstance(y, torch.Tensor):
                y = y.float()

            batch = (x, y)

        return super().step(batch, batch_idx)

    @property
    def default_loss_fn(self) -> Callable:
        if self.multilabel:
            return binary_cross_entropy
        return cross_entropy

    @staticmethod
    def default_pipeline() -> ClassificationDataPipeline:
        return ClassificationDataPipeline()
