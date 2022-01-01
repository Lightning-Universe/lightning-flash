from typing import Dict, Any, Optional, Callable, Union, List

import torchmetrics

from flash import Task
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

        # Attach the required collate function
        #adapter.set_state(CollateFn(partial(PyTorchForecastingAdapter._collate_fn, time_series_dataset._collate_fn)))

        return adapter
