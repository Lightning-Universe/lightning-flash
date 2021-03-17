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
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy

from flash.core.utils import bind_method
from flash.core.data import TaskDataPipeline
from flash.core.model import Task, predict_context


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
        """
        The training/validation/test step. Override for custom behavior.
        """
        x, y = batch
        if self.multilabel:
            y = y.float()
        y_hat = self.forward(x)
        output = {"y_hat": self.data_pipeline.before_uncollate(y_hat)}
        losses = {name: l_fn(y_hat, y) for name, l_fn in self.loss_fn.items()}
        logs = {}
        for name, metric in self.metrics.items():
            if isinstance(metric, pl.metrics.Metric):
                metric(output["y_hat"], y.long())
                logs[name] = metric  # log the metric itself if it is of type Metric
            else:
                logs[name] = metric(y_hat, y)
        logs.update(losses)
        if len(losses.values()) > 1:
            logs["total_loss"] = sum(losses.values())
            return logs["total_loss"], logs
        output["loss"] = list(losses.values())[0]
        output["logs"] = logs
        output["y"] = y
        return output

    @predict_context
    def predict(
        self,
        x: Any,
        batch_idx: Optional[int] = None,
        skip_collate_fn: bool = False,
        dataloader_idx: Optional[int] = None,
        data_pipeline: Optional[ClassificationDataPipeline] = None,
    ) -> Any:    
        if self.multilabel:
            bind_method(self.data_pipeline, 
                        lambda self, samples: samples.tolist(),
                        'after_uncollate'
            )
        return super().predict(x,
                      batch_idx,
                      skip_collate_fn,
                      dataloader_idx,
                      self.data_pipeline)

    @property
    def default_loss_fn(self) -> Callable:
        if self.multilabel:
            return binary_cross_entropy_with_logits
        return cross_entropy

    @staticmethod
    def default_pipeline() -> ClassificationDataPipeline:
        return ClassificationDataPipeline()
