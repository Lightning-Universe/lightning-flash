from typing import Any

import torch

from flash.core.model import Task
from flash.core.data import DataPipeline


class ClassificationDataPipeline(DataPipeline):
    def before_uncollate(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.softmax(batch, -1)

    def after_uncollate(self, samples: Any) -> Any:
        return torch.argmax(samples, -1).tolist()


class ClassificationTask(Task):
    @property
    def default_pipeline(self) -> ClassificationDataPipeline:
        return ClassificationDataPipeline()
