from typing import Any

import torch

from flash.core.data import DataPipeline
from flash.core.model import Task


class ClassificationDataPipeline(DataPipeline):
    def before_uncollate(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.softmax(batch, -1)

    def after_uncollate(self, samples: Any) -> Any:
        return torch.argmax(samples, -1).tolist()


class ClassificationTask(Task):
    @staticmethod
    def default_pipeline() -> ClassificationDataPipeline:
        return ClassificationDataPipeline()
