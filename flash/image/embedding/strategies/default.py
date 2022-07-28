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
import warnings
from typing import Any, Optional

from torch import nn

from flash.core.adapter import Adapter, AdapterTask
from flash.core.data.io.input import DataKeys
from flash.core.model import Task
from flash.core.registry import FlashRegistry
from flash.core.utilities.url_error import catch_url_error


class DefaultAdapter(Adapter):
    """The ``DefaultAdapter`` is an :class:`~flash.core.adapter.Adapter`."""

    required_extras: str = "image"

    def __init__(self, backbone: nn.Module):
        super().__init__()

        self.backbone = backbone

    @classmethod
    @catch_url_error
    def from_task(
        cls,
        task: AdapterTask,
        backbone: nn.Module,
        **kwargs,
    ) -> Adapter:
        adapter = cls(backbone)
        adapter.__dict__["_task"] = task
        return adapter

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        raise NotImplementedError(
            'Training an `ImageEmbedder` with `strategy="default"` is not supported. '
            "Use a different strategy instead."
        )

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        raise NotImplementedError(
            'Validation an `ImageEmbedder` with `strategy="default"` is not supported. '
            "Use a different strategy instead."
        )

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        raise NotImplementedError(
            'Testing an `ImageEmbedder` with `strategy="default"` is not supported. '
            "Use a different strategy instead."
        )

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        batch[DataKeys.PREDS] = Task.predict_step(
            self._task, (batch[DataKeys.INPUT]), batch_idx, dataloader_idx=dataloader_idx
        )
        return batch


def default(head: Optional[str] = None, loss_fn: Optional[str] = None, **kwargs):
    """Return `(None, None, [])` as loss function, head and hooks.

    Because default strategy only support prediction.
    """
    if head is not None:
        warnings.warn(f"default strategy has no heads. So given head({head}) is ignored.")

    if loss_fn is not None:
        warnings.warn(f"default strategy has no loss functions. So given loss_fn({loss_fn}) is ignored.")

    return None, None, []


def register_default_strategy(register: FlashRegistry):
    """Register default strategy to given ``FlashRegistry``."""
    register(default, name="default", adapter=DefaultAdapter)
