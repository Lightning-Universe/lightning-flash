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
from typing import Any, Callable, Dict, List, Optional, Union

import torchmetrics

from flash.core.adapter import AdapterTask
from flash.core.registry import FlashRegistry
from flash.core.utilities.types import LR_SCHEDULER_TYPE, OPTIMIZER_TYPE
from flash.tabular.forecasting.backbones import TABULAR_FORECASTING_BACKBONES


class TabularForecaster(AdapterTask):
    backbones: FlashRegistry = TABULAR_FORECASTING_BACKBONES

    def __init__(
        self,
        parameters: Dict[str, Any],
        backbone: str = "temporal_fusion_transformer",
        loss_fn: Optional[Callable] = None,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        metrics: Union[torchmetrics.Metric, List[torchmetrics.Metric]] = None,
        learning_rate: float = 3e-2,
        **backbone_kwargs
    ):

        self.save_hyperparameters()

        metadata = self.backbones.get(backbone, with_metadata=True)
        adapter = metadata["metadata"]["adapter"].from_task(
            self,
            parameters=parameters,
            backbone=backbone,
            loss_fn=loss_fn,
            metrics=metrics,
            **backbone_kwargs,
        )

        super().__init__(
            adapter,
            learning_rate=learning_rate,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
