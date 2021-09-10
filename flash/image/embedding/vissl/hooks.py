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
from typing import List

from pytorch_lightning.core.hooks import ModelHooks

from flash.core.utilities.imports import _VISSL_AVAILABLE

if _VISSL_AVAILABLE:
    from classy_vision.hooks.classy_hook import ClassyHook


class AdaptVISSLHooks(ModelHooks):
    def __init__(self, hooks: List[ClassyHook], task) -> None:
        super().__init__()

        self.hooks = hooks
        self.task = task

    def on_train_start(self) -> None:
        for hook in self.hooks:
            hook.on_start(self.task)

    # def on_train_end(self) -> None:
    #     for hook in self.hooks:
    #         hook.on_end()

    # def on_train_epoch_start(self) -> None:
    #     for hook in self.hooks:
    #         hook.on_phase_start()

    def on_train_epoch_end(self) -> None:
        for hook in self.hooks:
            hook.on_update(self.task)
            # hook.on_phase_end()

        self.task.iteration += 1

    # def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx) -> None:
    #     for hook in self.hooks:
    #         hook.on_step()

    # def on_after_backward(self) -> None:
    #     for hook in self.hooks:
    #         hook.on_backward()

    # def on_before_zero_grad(self, optimizer) -> None:
    #     for hook in self.hooks:
    #         hook.on_loss_and_meter()
