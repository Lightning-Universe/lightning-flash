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
from typing import List, Union

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import BaseFinetuning
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.optim import Optimizer


class NoFreeze(BaseFinetuning):
    def freeze_before_training(self, pl_module: LightningModule) -> None:
        pass

    def finetune_function(
        self,
        pl_module: LightningModule,
        epoch: int,
        optimizer: Optimizer,
        opt_idx: int,
    ) -> None:
        pass


class FlashBaseFinetuning(BaseFinetuning):
    """FlashBaseFinetuning can be used to create a custom Flash Finetuning Callback.

    Override :meth:`.finetune_function` to put your unfreeze logic.
    """

    def __init__(self, attr_names: Union[str, List[str]] = "backbone", train_bn: bool = True):
        """
        Args:
            attr_names: Name(s) of the module attributes of the model to be frozen.
            train_bn: Whether to train Batch Norm layer
        """
        super().__init__()

        self.attr_names = [attr_names] if isinstance(attr_names, str) else attr_names
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module: LightningModule) -> None:
        self.freeze_using_attr_names(pl_module, self.attr_names, train_bn=self.train_bn)

    def freeze_using_attr_names(self, pl_module, attr_names: List[str], train_bn: bool = True):
        for attr_name in attr_names:
            attr = getattr(pl_module, attr_name, None)
            if not attr or not isinstance(attr, nn.Module):
                MisconfigurationException(f"Your model must have a {attr} attribute")
            self.freeze(modules=attr, train_bn=train_bn)

    def finetune_function(self, pl_module: LightningModule, epoch: int, optimizer: Optimizer, opt_idx: int):
        pass


class Freeze(FlashBaseFinetuning):
    def finetune_function(
        self,
        pl_module: LightningModule,
        epoch: int,
        optimizer: Optimizer,
        opt_idx: int,
    ) -> None:
        pass


class FreezeUnfreeze(FlashBaseFinetuning):
    def __init__(self, attr_names: Union[str, List[str]] = "backbone", train_bn: bool = True, unfreeze_epoch: int = 10):
        super().__init__(attr_names, train_bn)
        self.unfreeze_epoch = unfreeze_epoch

    def finetune_function(
        self,
        pl_module: LightningModule,
        epoch: int,
        optimizer: Optimizer,
        opt_idx: int,
    ) -> None:
        if epoch != self.unfreeze_epoch:
            return
        modules = [getattr(pl_module, attr_name) for attr_name in self.attr_names]
        self.unfreeze_and_add_param_group(
            modules=modules,
            optimizer=optimizer,
            train_bn=self.train_bn,
        )


class UnfreezeMilestones(FlashBaseFinetuning):
    def __init__(
        self,
        attr_names: Union[str, List[str]] = "backbone",
        train_bn: bool = True,
        unfreeze_milestones: tuple = (5, 10),
        num_layers: int = 5,
    ):
        self.unfreeze_milestones = unfreeze_milestones
        self.num_layers = num_layers

        super().__init__(attr_names, train_bn)

    def finetune_function(
        self,
        pl_module: LightningModule,
        epoch: int,
        optimizer: Optimizer,
        opt_idx: int,
    ) -> None:
        backbone_modules = list(pl_module.backbone.modules())
        if epoch == self.unfreeze_milestones[0]:
            # unfreeze num_layers last layers
            self.unfreeze_and_add_param_group(
                modules=backbone_modules[-self.num_layers :],
                optimizer=optimizer,
                train_bn=self.train_bn,
            )

        elif epoch == self.unfreeze_milestones[1]:
            # unfreeze remaining layers
            self.unfreeze_and_add_param_group(
                modules=backbone_modules[: -self.num_layers],
                optimizer=optimizer,
                train_bn=self.train_bn,
            )


_DEFAULTS_FINETUNE_STRATEGIES = {
    "no_freeze": NoFreeze,
    "freeze": Freeze,
    "freeze_unfreeze": FreezeUnfreeze,
    "unfreeze_milestones": UnfreezeMilestones,
}


def instantiate_default_finetuning_callbacks(strategy):
    if not strategy or strategy not in _DEFAULTS_FINETUNE_STRATEGIES:
        raise MisconfigurationException(
            f"a strategy should be provided. Use {list(_DEFAULTS_FINETUNE_STRATEGIES)} or provide a callback"
            " instance of `flash.core.finetuning.FlashBaseFinetuning`. Found {strategy} "
        )
    if isinstance(strategy, str):
        strategy = strategy.lower()
        if strategy in _DEFAULTS_FINETUNE_STRATEGIES:
            return [_DEFAULTS_FINETUNE_STRATEGIES[strategy]()]
    return []
