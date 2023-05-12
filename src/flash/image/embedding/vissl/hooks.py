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
from typing import Any, List

import torch
from pytorch_lightning.core.hooks import ModelHooks

import flash
from flash.core.utilities.compatibility import accelerator_connector
from flash.core.utilities.imports import _VISSL_AVAILABLE

if _VISSL_AVAILABLE:
    from classy_vision.hooks.classy_hook import ClassyHook
else:

    class ClassyHook:
        _noop = object


class TrainingSetupHook(ClassyHook):
    on_start = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_step = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop
    on_update = ClassyHook._noop
    on_forward = ClassyHook._noop

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def on_start(self, task: "flash.image.embedding.vissl.adapter.MockVISSLTask") -> None:
        lightning_module = task.vissl_adapter.adapter_task
        task.device = lightning_module.device

        # get around vissl distributed training by setting MockTask flags
        num_nodes = lightning_module.trainer.num_nodes
        accelerators_ids = getattr(
            lightning_module.trainer,
            "device_ids",
            getattr(accelerator_connector(lightning_module.trainer), "parallel_device_ids", None),
        )
        accelerator_per_node = len(accelerators_ids) if accelerators_ids is not None else 1
        task.world_size = num_nodes * accelerator_per_node

        if lightning_module.trainer.max_epochs is None:
            lightning_module.trainer.max_epochs = 1

        task.max_iteration = lightning_module.trainer.max_epochs * lightning_module.trainer.num_training_batches


class SimCLRTrainingSetupHook(TrainingSetupHook):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def on_start(self, task: "flash.image.embedding.vissl.adapter.MockVISSLTask") -> None:
        super().on_start(task)

        lightning_module = task.vissl_adapter.adapter_task

        # specific to simclr in VISSL
        task.loss.info_criterion.buffer_params.effective_batch_size = (
            task.world_size * 2 * lightning_module.trainer.datamodule.batch_size
        )
        task.loss.info_criterion.buffer_params.world_size = task.world_size

        task.loss.info_criterion.precompute_pos_neg_mask()

        # Cast the loss to the correct device / dtype
        task.loss.to(lightning_module.device, lightning_module.dtype)


class AdaptVISSLHooks(ModelHooks):
    def __init__(self, hooks: List[ClassyHook], task) -> None:
        super().__init__()

        self.hooks = hooks
        self.task = task

    def on_train_start(self) -> None:
        for hook in self.hooks:
            hook.on_start(self.task)

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int, *args) -> None:
        self.task.iteration += 1

    def on_train_epoch_end(self) -> None:
        for hook in self.hooks:
            hook.on_update(self.task)
