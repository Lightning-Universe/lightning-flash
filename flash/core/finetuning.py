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
from functools import partial
from typing import Iterable, Optional, Tuple, Union

from pytorch_lightning.callbacks import BaseFinetuning
from pytorch_lightning.utilities.enums import LightningEnum
from torch.nn import Module
from torch.optim import Optimizer

from flash.core.registry import FlashRegistry


class FinetuningStrategies(LightningEnum):
    """The ``FinetuningStrategies`` enum contains the keys that are used internally by the ``FlashBaseFinetuning``
    when choosing the strategy to perform."""

    NO_FREEZE = "no_freeze"
    FREEZE = "freeze"
    FREEZE_UNFREEZE = "freeze_unfreeze"
    UNFREEZE_MILESTONES = "unfreeze_milestones"
    CUSTOM = "custom"

    # TODO: Create a FlashEnum class???
    def __hash__(self) -> int:
        return hash(self.value)


class FlashBaseFinetuning(BaseFinetuning):
    """FlashBaseFinetuning can be used to create a custom Flash Finetuning Callback."""

    def __init__(
        self,
        strategy_key: FinetuningStrategies,
        strategy_metadata: Optional[Union[int, Tuple[int, int]]] = None,
        train_bn: bool = True,
    ):
        """
        Args:
            strategy_key: The finetuning strategy to be used. (Check :meth"`Trainer.finetune` for the available
             strategies.)
            strategy_metadata: Data that accompanies certain finetuning strategies like epoch number or number of
             layers.
            attr_names: Name(s) of the module attributes of the model to be frozen.
            train_bn: Whether to train Batch Norm layer
        """
        super().__init__()

        self.strategy: FinetuningStrategies = strategy_key
        self.strategy_metadata: Optional[Union[int, Tuple[int, int]]] = strategy_metadata
        self.train_bn: bool = train_bn

    def freeze_before_training(self, pl_module: Union[Module, Iterable[Union[Module, Iterable]]]) -> None:
        if self.strategy != FinetuningStrategies.NO_FREEZE:
            get_backbone_for_finetuning = getattr(pl_module, "get_backbone_for_finetuning", None)
            if get_backbone_for_finetuning is None:
                raise AttributeError(
                    "Lightning Module missing instance method 'get_backbone_for_finetuning'."
                    "Please, implement the method which returns NoneType or a Module or an Iterable of Modules."
                )
            modules = get_backbone_for_finetuning()
            if modules is not None:
                if isinstance(modules, Module):
                    modules = [modules]
                self.freeze(modules=modules, train_bn=self.train_bn)

    def _freeze_unfreeze_function(
        self,
        pl_module: Union[Module, Iterable[Union[Module, Iterable]]],
        epoch: int,
        optimizer: Optimizer,
        opt_idx: int,
        strategy_metadata: int,
    ):
        unfreeze_epoch: int = strategy_metadata
        if epoch != unfreeze_epoch:
            return

        get_backbone_for_finetuning = getattr(pl_module, "get_backbone_for_finetuning", None)
        if get_backbone_for_finetuning is None:
            raise AttributeError(
                "Lightning Module missing instance method 'get_backbone_for_finetuning'."
                "Please, implement the method which returns NoneType or a Module or an Iterable of Modules."
            )

        modules = get_backbone_for_finetuning()
        if modules is not None:
            self.unfreeze_and_add_param_group(
                modules=modules,
                optimizer=optimizer,
                train_bn=self.train_bn,
            )

    def _unfreeze_milestones_function(
        self,
        pl_module: Union[Module, Iterable[Union[Module, Iterable]]],
        epoch: int,
        optimizer: Optimizer,
        opt_idx: int,
        strategy_metadata: Tuple[Tuple[int, int], int],
    ):
        unfreeze_milestones: Tuple[int, int] = strategy_metadata[0]
        num_layers: int = strategy_metadata[1]

        get_backbone_for_finetuning = getattr(pl_module, "get_backbone_for_finetuning", None)
        if get_backbone_for_finetuning is None:
            raise AttributeError(
                "Lightning Module missing instance method 'get_backbone_for_finetuning'."
                "Please, implement the method which returns NoneType or a Module or an Iterable of Modules."
            )
        modules = get_backbone_for_finetuning()
        if modules is not None:
            if epoch == unfreeze_milestones[0]:
                # unfreeze num_layers last layers

                backbone_modules = BaseFinetuning.flatten_modules(modules=modules)[-num_layers:]
                self.unfreeze_and_add_param_group(
                    modules=backbone_modules,
                    optimizer=optimizer,
                    train_bn=self.train_bn,
                )
            elif epoch == unfreeze_milestones[1]:
                # unfreeze remaining layers
                backbone_modules = BaseFinetuning.flatten_modules(modules=modules)[:-num_layers]
                self.unfreeze_and_add_param_group(
                    modules=backbone_modules,
                    optimizer=optimizer,
                    train_bn=self.train_bn,
                )

    def finetune_function(
        self,
        pl_module: Union[Module, Iterable[Union[Module, Iterable]]],
        epoch: int,
        optimizer: Optimizer,
        opt_idx: int,
    ):
        if self.strategy == FinetuningStrategies.FREEZE_UNFREEZE:
            self._freeze_unfreeze_function(pl_module, epoch, optimizer, opt_idx, self.strategy_metadata)
        elif self.strategy == FinetuningStrategies.UNFREEZE_MILESTONES:
            self._unfreeze_milestones_function(pl_module, epoch, optimizer, opt_idx, self.strategy_metadata)
        elif self.strategy == FinetuningStrategies.CUSTOM:
            finetune_backbone = getattr(pl_module, "finetune_backbone", None)
            if finetune_backbone is None:
                raise AttributeError(
                    "Lightning Module missing instance method 'finetune_backbone'."
                    "Please, implement the method which performs the necessary finetuning of the backbone."
                )
            finetune_backbone(epoch, optimizer, opt_idx)
        else:
            pass


# Used for properly verifying input and providing neat and helpful error messages for users.
_DEFAULTS_FINETUNE_STRATEGIES = [
    "custom",
    "no_freeze",
    "freeze",
    "freeze_unfreeze",
    "unfreeze_milestones",
]

_FINETUNING_STRATEGIES_REGISTRY = FlashRegistry("finetuning_strategies")
for strategy in FinetuningStrategies:
    _FINETUNING_STRATEGIES_REGISTRY(
        name=strategy.value,
        fn=partial(FlashBaseFinetuning, strategy_key=strategy),
    )
