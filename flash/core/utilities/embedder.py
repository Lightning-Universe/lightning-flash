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
from typing import Any

from pytorch_lightning import LightningModule

from flash.core.model import Task


class StopForward(Exception):
    pass


class Embedder(Task):
    def __init__(self, model: LightningModule, layer: str):
        super().__init__()

        self.model = model
        self.layer = layer

        self._module, self._hook = self._make_hook()
        self._handle = None
        self._out = None

    def _make_hook(self):
        def hook(_, __, output):
            self._out = output
            raise StopForward

        available_layers = {"output", ""}

        if self.layer in available_layers:
            return None, None

        for name, module in self.model.named_modules():
            available_layers.add(name)
            if name == self.layer:
                return module, hook

        raise ValueError(
            "The requested layer is not available in `model.named_modules`. The available layers are: "
            f"{', '.join(available_layers)}."
        )

    def _register_hook(self):
        if self._module is not None:
            self._handle = self._module.register_forward_hook(self._hook)

    def _remove_hook(self):
        if self._handle is not None:
            self._handle.remove()
        self._handle = None

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        raise NotImplementedError("Training an `Embedder` is not supported.")

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        raise NotImplementedError("Validating an `Embedder` is not supported.")

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        raise NotImplementedError("Testing an `Embedder` is not supported.")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        try:
            self._register_hook()
            return self.model.predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        except StopForward:
            return self._out
        finally:
            self._remove_hook()
            self._out = None
