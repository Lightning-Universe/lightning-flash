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
from pytorch_lightning import LightningModule

from flash.core.adapter import AdapterTask
from flash.core.integrations.pytorch_forecasting.adapter import PyTorchForecastingAdapter
from flash.core.integrations.pytorch_forecasting.backbones import PYTORCH_FORECASTING_BACKBONES
from flash.core.registry import FlashRegistry
from flash.core.utilities.types import LR_SCHEDULER_TYPE, OPTIMIZER_TYPE


class TabularForecaster(AdapterTask):

    backbones: FlashRegistry = FlashRegistry("backbones") + PYTORCH_FORECASTING_BACKBONES
    required_extras: str = "tabular"

    def __init__(
        self,
        parameters: Dict[str, Any],
        backbone: str,
        backbone_kwargs: Optional[Dict[str, Any]] = None,
        loss_fn: Optional[Callable] = None,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        metrics: Union[torchmetrics.Metric, List[torchmetrics.Metric]] = None,
        learning_rate: Optional[float] = None,
    ):
        self.save_hyperparameters()

        if backbone_kwargs is None:
            backbone_kwargs = {}

        metadata = self.backbones.get(backbone, with_metadata=True)
        adapter = metadata["metadata"]["adapter"].from_task(
            self,
            parameters=parameters,
            backbone=backbone,
            backbone_kwargs=backbone_kwargs,
            loss_fn=loss_fn,
            metrics=metrics,
        )

        super().__init__(
            adapter,
            learning_rate=learning_rate,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

    @property
    def pytorch_forecasting_model(self) -> LightningModule:
        """This property provides access to the ``LightningModule`` object that is wrapped by Flash for backbones
        provided by PyTorch Forecasting.

        This can be used with
        :func:`~flash.core.integrations.pytorch_forecasting.transforms.convert_predictions` to access the visualization
        features built in to PyTorch Forecasting.
        """
        if not isinstance(self.adapter, PyTorchForecastingAdapter):
            raise AttributeError(
                "The `pytorch_forecasting_model` attribute can only be accessed for backbones provided by PyTorch "
                "Forecasting."
            )
        return self.adapter.backbone
