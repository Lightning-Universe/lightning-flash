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
from classy_vision.hooks.classy_hook import ClassyHook

from flash.core.adapter import Adapter
from flash.core.data.data_source import DefaultDataKeys
from flash.core.model import Task
from flash.core.utilities.imports import _VISSL_AVAILABLE
from flash.core.utilities.url_error import catch_url_error

if _VISSL_AVAILABLE:
    from classy_vision.losses import ClassyLoss
    from vissl.config.attr_dict import AttrDict
    from vissl.models.base_ssl_model import BaseSSLMultiInputOutputModel

    from flash.image.embedding.vissl.hooks import AdaptVISSLHooks


class MockVISSLTask:
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
        self.last_batch = AttrDict({"sample": AttrDict({"input": None})})


class VISSLAdapter(Adapter, AdaptVISSLHooks):
    """The ``VISSLAdapter`` is an :class:`~flash.core.adapter.Adapter` for integrating with VISSL."""

    required_extras: str = "image"

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        loss_fn: ClassyLoss,
        embedding_dim: int,
        hooks: List[ClassyHook],
        **kwargs,
    ) -> None:

        Adapter.__init__(self)

        self.model_config = self.get_model_config_template()
        self.optimizer_config = AttrDict({})

        self.backbone = backbone
        self.head = [head] if not isinstance(head, list) else head
        self.loss_fn = loss_fn
        self.embedding_dim = embedding_dim
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

        self.vissl_task = MockVISSLTask(
            self, self.loss_fn, self.task_config, self.vissl_base_model
        )

        AdaptVISSLHooks.__init__(self, hooks=hooks, task=self.vissl_task)

        # task.config["MODEL"], task.config["OPTIMIZER"]
        # patch task.loss.momentum teacher, deepcopy from trunk
        # mock task only needs to be passed for hooks, avoid all
        # vissl_task.base_model is vissl_trunk
        #
        # make sure momentum_teacher is not updated with backprop, only needs to
        # be updated with momentum hook
        # detach on teacher output or torch.no_grad()?

        # Loss config is as follows:
        # LOSS:
        #   name: loss_name
        #   loss_name:
        #       param1:
        #       param2:
        #       ...

    @classmethod
    @catch_url_error
    def from_task(
        cls,
        task: Task,
        loss_fn: ClassyLoss,
        backbone: nn.Module,
        embedding_dim: int,
        head: Union[nn.Module, List[nn.Module]],
        hooks: List[ClassyHook],
        **kwargs,
    ) -> Adapter:
        result = cls(
            backbone=backbone,
            head=head,
            loss_fn=loss_fn,
            embedding_dim=embedding_dim,
            hooks=hooks,
            **kwargs,
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

    def forward(self, batch) -> Any:
        model_output = self.vissl_base_model(batch)

        # vissl-specific
        if len(model_output) == 1:
            model_output = model_output[0]

        return model_output

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        out = self(batch[DefaultDataKeys.INPUT])
        self.task.last_batch["sample"]["input"] = batch[DefaultDataKeys.INPUT]

        # call forward hook from VISSL (momentum updates)
        for hook in self.hooks:
            hook.on_forward(self.vissl_task)

        loss = self.loss_fn(out, target=None)
        self.adapter_task.log_dict({"train_loss": loss.item()})

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        out = self(batch[DefaultDataKeys.INPUT])
        self.task.last_batch["sample"]["input"] = batch[DefaultDataKeys.INPUT]

        loss = self.loss_fn(out, target=None)
        self.adapter_task.log_dict({"val_loss": loss})

        return loss

    def test_step(self, batch: Any, batch_idx: int) -> None:
        # vissl_input, target = batch
        # out = self(vissl_input)

        # # out can be torch.Tensor/List target is torch.Tensor
        # loss = self.vissl_loss(out, target)

        # # TODO: log
        # # TODO: Include call to ClassyHooks during training
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        # TODO: return embedding here
        pass
