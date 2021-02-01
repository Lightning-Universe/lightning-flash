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
import inspect
from typing import List, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import BaseFinetuning
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.optim import Optimizer

_EXCLUDE_PARAMTERS = ("self", "args", "kwargs")


class FlashBaseFinetuning(BaseFinetuning):

    def __init__(self, attr_names: Union[str, List[str]] = "backbone", train_bn: bool = True):
        r"""

        FlashBaseFinetuning can be used to create a custom Flash Finetuning Callback.

        Override ``finetunning_function`` to put your unfreeze logic.

        Args:
            attr_names: Name(s) of the module attributes of the model to be frozen.

            train_bn: Wether to train Batch Norm layer

        """

        self.attr_names = [attr_names] if isinstance(attr_names, str) else attr_names
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module: pl.LightningModule) -> None:
        self.freeze_using_attr_names(pl_module, self.attr_names, train_bn=self.train_bn)

    @staticmethod
    def freeze_using_attr_names(pl_module, attr_names: List[str], train_bn: bool = True):
        for attr_name in attr_names:
            attr = getattr(pl_module, attr_name, None)
            if attr is None or not isinstance(attr, nn.Module):
                MisconfigurationException(f"Your model must have a {attr} attribute")
            BaseFinetuning.freeze(module=attr, train_bn=train_bn)


class FreezeUnfreeze(FlashBaseFinetuning):

    def __init__(self, attr_names: Union[str, List[str]] = "backbone", train_bn: bool = True, unfreeze_epoch: int = 10):
        super().__init__(attr_names, train_bn)
        self.unfreeze_epoch = unfreeze_epoch

    def freeze_before_training(self, pl_module: pl.LightningModule) -> None:
        self.freeze_using_attr_names(pl_module, self.attr_names, train_bn=self.train_bn)

    def finetunning_function(
        self,
        pl_module: pl.LightningModule,
        epoch: int,
        optimizer: Optimizer,
        opt_idx: int,
    ) -> None:
        if epoch == self.unfreeze_epoch:
            modules = []
            for attr_name in self.attr_names:
                modules.append(getattr(pl_module, attr_name))

            self.unfreeze_and_add_param_group(
                module=modules,
                optimizer=optimizer,
                train_bn=self.train_bn,
            )


class UnfreezeMilestones(FlashBaseFinetuning):

    def __init__(
        self,
        attr_names: Union[str, List[str]] = "backbone",
        train_bn: bool = True,
        unfreeze_milestones: tuple = (5, 10),
        num_layers: int = 5
    ):
        self.unfreeze_milestones = unfreeze_milestones
        self.num_layers = num_layers

        super().__init__(attr_names, train_bn)

    def freeze_before_training(self, pl_module: pl.LightningModule) -> None:
        self.freeze_using_attr_names(pl_module, self.attr_names, train_bn=self.train_bn)

    def finetunning_function(
        self,
        pl_module: pl.LightningModule,
        epoch: int,
        optimizer: Optimizer,
        opt_idx: int,
    ) -> None:
        backbone_modules = list(pl_module.backbone.modules())
        if epoch == self.unfreeze_milestones[0]:
            # unfreeze 5 last layers
            # TODO last N layers should be parameter
            self.unfreeze_and_add_param_group(
                module=backbone_modules[-self.num_layers:],
                optimizer=optimizer,
                train_bn=self.train_bn,
            )

        elif epoch == self.unfreeze_milestones[1]:
            # unfreeze remaining layers
            # TODO last N layers should be parameter
            self.unfreeze_and_add_param_group(
                module=backbone_modules[:-self.num_layers],
                optimizer=optimizer,
                train_bn=self.train_bn,
            )


_DEFAULTS_FINETUNE_STRATEGIES = {
    "no_freeze": BaseFinetuning,
    "freeze": FlashBaseFinetuning,
    "freeze_unfreeze": FreezeUnfreeze,
    "unfreeze_milestones": UnfreezeMilestones
}


def instantiate_default_finetuning_callbacks(strategy):
    if strategy is None:
        strategy = "no_freeze"
        rank_zero_warn("strategy is None. Setting strategy to `no_freeze` by default.", UserWarning)
    if isinstance(strategy, str):
        strategy = strategy.lower()
        if strategy in _DEFAULTS_FINETUNE_STRATEGIES:
            return [_DEFAULTS_FINETUNE_STRATEGIES[strategy]()]
        raise MisconfigurationException(
            f"strategy should be within {list(_DEFAULTS_FINETUNE_STRATEGIES)}"
            f". Found {strategy}"
        )
    return []
