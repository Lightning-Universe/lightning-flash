import warnings
from typing import Any, Optional

import torch

from flash.core.adapter import Adapter, AdapterTask
from flash.core.data.io.input import DataKeys
from flash.core.model import Task
from flash.core.registry import FlashRegistry
from flash.core.utilities.url_error import catch_url_error


class DefaultAdapter(Adapter):
    """The ``DefaultAdapter`` is an :class:`~flash.core.adapter.Adapter`."""

    required_extras: str = "image"

    def __init__(self, backbone: torch.nn.Module, head: torch.nn.Module):
        super().__init__()

        self.backbone = backbone
        self.head = head

    @classmethod
    @catch_url_error
    def from_task(
        cls,
        *args,
        task: AdapterTask,
        backbone: torch.nn.Module,
        head: torch.nn.Module,
        **kwargs,
    ) -> Adapter:
        adapter = cls(backbone, head)
        adapter.__dict__["_task"] = task
        return adapter

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        raise NotImplementedError(
            "Training a `ImageEmbedder` with `DefaultAdapter` is not supported. Use a different strategy instead."
        )

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DataKeys.INPUT], batch[DataKeys.TARGET])
        return Task.validation_step(self._task, batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DataKeys.INPUT], batch[DataKeys.TARGET])
        return Task.test_step(self._task, batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        batch[DataKeys.PREDS] = Task.predict_step(
            self._task, (batch[DataKeys.INPUT]), batch_idx, dataloader_idx=dataloader_idx
        )
        return batch

    def forward(self, x) -> torch.Tensor:
        # TODO: Resolve this hack
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return self.backbone(x)


def default(head: Optional[str] = None, loss_fn: Optional[str] = None, **kwargs):
    if head is not None:
        warnings.warn(f"default strategy has no heads. So given head({head}) is ignored.")

    if loss_fn is not None:
        warnings.warn(f"default strategy has no loss functions. So given loss_fn({loss_fn}) is ignored.")

    return None, None, []


def register_default_strategy(register: FlashRegistry):
    register(default, name="default", adapter=DefaultAdapter)
