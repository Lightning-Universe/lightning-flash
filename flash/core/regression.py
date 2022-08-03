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
from typing import Callable, Mapping, Optional, Sequence, Union

import torch.nn.functional as F
import torchmetrics
from torch import Tensor

from flash.core.adapter import AdapterTask
from flash.core.model import Task


class RegressionMixin:
    @staticmethod
    def _build(
        loss_fn: Optional[Callable] = None,
        metrics: Union[torchmetrics.Metric, Mapping, Sequence, None] = None,
    ):
        metrics = metrics or torchmetrics.MeanSquaredError()
        loss_fn = loss_fn or F.mse_loss

        return metrics, loss_fn

    def to_metrics_format(self, x: Tensor) -> Tensor:
        return x


class RegressionTask(Task, RegressionMixin):
    def __init__(
        self,
        *args,
        loss_fn: Optional[Callable] = None,
        metrics: Union[torchmetrics.Metric, Mapping, Sequence, None] = None,
        **kwargs,
    ) -> None:

        metrics, loss_fn = RegressionMixin._build(loss_fn, metrics)

        super().__init__(
            *args,
            loss_fn=loss_fn,
            metrics=metrics,
            **kwargs,
        )


class RegressionAdapterTask(AdapterTask, RegressionMixin):
    def __init__(
        self,
        *args,
        loss_fn: Optional[Callable] = None,
        metrics: Union[torchmetrics.Metric, Mapping, Sequence, None] = None,
        **kwargs,
    ) -> None:
        metrics, loss_fn = RegressionMixin._build(loss_fn, metrics)

        super().__init__(
            *args,
            loss_fn=loss_fn,
            metrics=metrics,
            **kwargs,
        )
