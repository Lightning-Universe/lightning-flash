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
from typing import Any, List, Union

import torch
import torch.nn as nn

from flash.core.adapter import Adapter
from flash.core.data.io.input import DataKeys
from flash.core.model import Task
from flash.core.utilities.imports import _VISSL_AVAILABLE
from flash.image.embedding.vissl.hooks import AdaptVISSLHooks

if _VISSL_AVAILABLE:
    from classy_vision.hooks.classy_hook import ClassyHook
    from classy_vision.losses import ClassyLoss
    from vissl.config.attr_dict import AttrDict
    from vissl.models.base_ssl_model import BaseSSLMultiInputOutputModel
else:
    ClassyLoss = object
    ClassyHook = object


class MockVISSLTask:
    """Mock task class from VISSL to support loss, configs, base_model, last batch etc."""

    def __init__(self, vissl_adapter, vissl_loss, task_config, vissl_model) -> None:
        self.vissl_adapter = vissl_adapter
        self.loss = vissl_loss
        self.config = task_config
        self.base_model = vissl_model
        self.model = self.base_model  # set by property in ClassyTask

        # set using trainingsetuphook
        self.device = None

        self.iteration = 0
        self.max_iteration = 1  # set by training setup hook

        # set for momentum teacher based hooks
        self.last_batch = AttrDict({"sample": AttrDict({"input": None, "data_momentum": None})})


class VISSLAdapter(Adapter, AdaptVISSLHooks):
    """The ``VISSLAdapter`` is an :class:`~flash.core.adapter.Adapter` for integrating with VISSL.

    Also inherits from ``AdaptVISSLHooks`` to support VISSL hooks.
    """

    required_extras: str = "image"

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        loss_fn: ClassyLoss,
        hooks: List[ClassyHook],
    ) -> None:

        Adapter.__init__(self)

        self.model_config = self.get_model_config_template()
        self.optimizer_config = AttrDict({})

        self.backbone = backbone
        self.head = [head] if not isinstance(head, list) else head
        self.loss_fn = loss_fn
        self.hooks = hooks

        self.model_config.TRUNK = self.backbone.model_config.TRUNK
        self.model_config.HEAD = self.head[0].model_config.HEAD
        self.task_config = AttrDict(
            {
                "MODEL": self.model_config,
                "OPTIMIZER": self.optimizer_config,
                "LOSS": AttrDict(
                    {
                        "name": self.loss_fn.loss_name,
                        self.loss_fn.loss_name: self.loss_fn.loss_config,
                    }
                ),
            }
        )

        self.vissl_base_model = BaseSSLMultiInputOutputModel(self.model_config, self.optimizer_config)
        # patch backbone and head
        self.vissl_base_model.trunk = backbone
        self.vissl_base_model.heads = nn.ModuleList(self.head)

        self.vissl_task = MockVISSLTask(self, self.loss_fn, self.task_config, self.vissl_base_model)

        AdaptVISSLHooks.__init__(self, hooks=hooks, task=self.vissl_task)

    @classmethod
    def from_task(
        cls,
        task: Task,
        loss_fn: ClassyLoss,
        backbone: nn.Module,
        head: Union[nn.Module, List[nn.Module]],
        hooks: List[ClassyHook],
    ) -> Adapter:
        result = cls(
            backbone=backbone,
            head=head,
            loss_fn=loss_fn,
            hooks=hooks,
        )

        result.__dict__["adapter_task"] = task

        return result

    @staticmethod
    def get_model_config_template():
        cfg = AttrDict(
            {
                "SINGLE_PASS_EVERY_CROP": False,
                "INPUT_TYPE": "rgb",
                "MULTI_INPUT_HEAD_MAPPING": [],
                "TRUNK": AttrDict({}),
                "HEAD": AttrDict(
                    {
                        "PARAMS": [],
                        "BATCHNORM_EPS": 1e-5,
                        "BATCHNORM_MOMENTUM": 0.1,
                        "PARAMS_MULTIPLIER": 1.0,
                    }
                ),
                "FEATURE_EVAL_SETTINGS": AttrDict(
                    {
                        "EVAL_MODE_ON": False,
                        "EXTRACT_TRUNK_FEATURES_ONLY": False,
                    }
                ),
                "_MODEL_INIT_SEED": 0,
                "ACTIVATION_CHECKPOINTING": AttrDict(
                    {
                        "USE_ACTIVATION_CHECKPOINTING": False,
                        "NUM_ACTIVATION_CHECKPOINTING_SPLITS": 2,
                    }
                ),
            }
        )

        return cfg

    def forward(self, batch: torch.Tensor) -> Any:
        return self.vissl_base_model.trunk(batch, [])[0]

    def ssl_forward(self, batch) -> Any:
        model_output = self.vissl_base_model(batch)

        # vissl-specific
        if len(model_output) == 1:
            model_output = model_output[0]

        return model_output

    def shared_step(self, batch: Any, train: bool = True) -> Any:
        out = self.ssl_forward(batch[DataKeys.INPUT])

        # for moco and dino
        self.task.last_batch["sample"]["input"] = batch[DataKeys.INPUT]
        if "data_momentum" in batch.keys():
            self.task.last_batch["sample"]["data_momentum"] = [batch["data_momentum"]]

        if train:
            # call forward hook from VISSL (momentum updates)
            for hook in self.hooks:
                hook.on_forward(self.vissl_task)

        loss = self.loss_fn(out, target=None)

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        loss = self.shared_step(batch)
        self.adapter_task.log_dict({"train_loss": loss.item()})

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        loss = self.shared_step(batch, train=False)
        self.adapter_task.log_dict({"val_loss": loss})

        return loss

    def test_step(self, batch: Any, batch_idx: int) -> None:
        loss = self.shared_step(batch, train=False)
        self.adapter_task.log_dict({"test_loss": loss})

        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        input_image = batch[DataKeys.INPUT]

        return self(input_image)
