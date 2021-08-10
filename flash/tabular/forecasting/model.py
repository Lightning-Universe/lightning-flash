from typing import Union, Optional, Tuple, Dict, Callable, Type, Any, Mapping, Sequence, List

from pytorch_forecasting import BaseModel, QuantileLoss, SMAPE
from torch.optim import Optimizer

from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.process import Deserializer, Postprocess
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _TORCH_AVAILABLE

if _TORCH_AVAILABLE:
    import torch
    import torchmetrics
    from torch import nn
    from torch.optim.lr_scheduler import _LRScheduler
else:
    _LRScheduler = object

from flash import Task, Serializer, Preprocess
from flash.tabular.forecasting import (
    TabularForecastingData,
    TABULAR_FORECASTING_BACKBONES
)


class TabularForecaster(Task):
    backbones: FlashRegistry = TABULAR_FORECASTING_BACKBONES

    def __init__(
            self,
            tabular_forecasting_data: TabularForecastingData,
            backbone: Union[str, Tuple[nn.Module, int]] = "temporal_fusion_transformer",
            backbone_kwargs: Optional[Dict] = None,
            loss_fn: Optional[Callable] = None,
            optimizer: Union[Type[torch.optim.Optimizer], torch.optim.Optimizer, str] = torch.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            scheduler: Optional[Union[Type[_LRScheduler], str, _LRScheduler]] = None,
            scheduler_kwargs: Optional[Dict[str, Any]] = None,
            metrics: Union[torchmetrics.Metric, Mapping, Sequence, None] = None,
            learning_rate: float = 1e-2,
            **task_kwargs
    ):

        super().__init__(
            model=None,
            loss_fn=None,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            metrics=metrics,
            learning_rate=learning_rate,
            **task_kwargs
        )

        self.save_hyperparameters()

        if not backbone_kwargs:
            backbone_kwargs = {}

        if isinstance(backbone, tuple):
            self.backbone = backbone
        else:
            self.backbone = self.backbones.get(backbone)(
                tabular_forecasting_data=tabular_forecasting_data,
                **backbone_kwargs
            )
        self.model = self.backbone

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return self.model.training_step(batch, batch_idx)

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return self.model.validation_step(batch, batch_idx)

    def configure_optimizers(self) -> Union[Optimizer, Tuple[List[Optimizer], List[_LRScheduler]]]:
        return self.model.configure_optimizers()



    # More hooks to map

    @classmethod
    def from_data(cls, tabular_forecasting_data: TabularForecastingData, **kwargs):
        return cls(
            tabular_forecasting_data=tabular_forecasting_data,
            backbone_kwargs={"loss": SMAPE()},
            **kwargs
        )
