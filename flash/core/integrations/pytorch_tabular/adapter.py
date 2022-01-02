from typing import Dict, Any, Optional, Callable, Union, List

import torchmetrics

from flash import Task, DataKeys
from flash.core.adapter import Adapter


class PytorchTabularAdapter(Adapter):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone

    @classmethod
    def from_task(
            cls,
            task: Task,
            task_type,
            parameters: Dict[str, Any],
            backbone: str,
            backbone_kwargs: Optional[Dict[str, Any]] = None,
            loss_fn: Optional[Callable] = None,
            metrics: Optional[Union[torchmetrics.Metric, List[torchmetrics.Metric]]] = None,
    ) -> Adapter:
        # Remove the single row of data from the parameters to reconstruct the `time_series_dataset`

        backbone_kwargs = backbone_kwargs or {}

        adapter = cls(task.backbones.get(backbone)(task_type=task_type, parameters=parameters, **backbone_kwargs))

        return adapter

    @staticmethod
    def convert_batch(batch):
        return {
            "target": batch[DataKeys.TARGET].reshape(-1, 1),
            "continuous": batch[DataKeys.INPUT][1],
            "categorical": batch[DataKeys.INPUT][0],
        }

    def training_step(self, batch, batch_idx) -> Any:
        return self.backbone.training_step(self.convert_batch(batch), batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.backbone.validation_step(self.convert_batch(batch), batch_idx)

    def test_step(self, batch, batch_idx):
        return self.backbone.validation_step(self.convert_batch(batch), batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        batch[DataKeys.PREDS] = self(self.convert_batch(batch))
        return batch
