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

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import BaseFinetuning
from pytorch_lightning.utilities.enums import LightningEnum
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.nn import Module
from torch.optim import Optimizer

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _PL_AVAILABLE


class FinetuningStrategies(LightningEnum):
    """The ``FinetuningStrategies`` enum contains the keys that are used internally by the ``FlashBaseFinetuning``
    when choosing the strategy to perform."""

    NO_FREEZE = "no_freeze"
    FREEZE = "freeze"
    FREEZE_UNFREEZE = "freeze_unfreeze"
    UNFREEZE_MILESTONES = "unfreeze_milestones"

    # TODO: Create a FlashEnum class???
    def __hash__(self) -> int:
        return hash(self.value)


class FlashBaseFinetuning(BaseFinetuning):
    """FlashBaseFinetuning can be used to create a custom Flash Finetuning Callback."""

    def __init__(
        self,
        strategy_key: FinetuningStrategies,
        strategy_metadata: Optional[Union[int, Tuple[Tuple[int, int], int]]] = None,
        train_bn: bool = True,
    ):
        """
        Args:
            strategy_key: The finetuning strategy to be used. See :meth:`~flash.core.trainer.Trainer.finetune`
                        for the available strategies.
            strategy_metadata: Data that accompanies certain finetuning strategies like epoch number or number of
                        layers.
            attr_names: Name(s) of the module attributes of the model to be frozen.
            train_bn: Whether to train Batch Norm layer
        """
        super().__init__()

        self.strategy: FinetuningStrategies = strategy_key
        self.strategy_metadata: Optional[Union[int, Tuple[Tuple[int, int], int]]] = strategy_metadata
        self.train_bn: bool = train_bn

        if self.strategy == "freeze_unfreeze" and not isinstance(self.strategy_metadata, int):
            raise MisconfigurationException(
                "`freeze_unfreeze` stratgey only accepts one integer denoting the epoch number to switch."
            )
        if self.strategy == "unfreeze_milestones" and not (
            isinstance(self.strategy_metadata, Tuple)
            and isinstance(self.strategy_metadata[0], Tuple)
            and isinstance(self.strategy_metadata[1], int)
            and isinstance(self.strategy_metadata[0][0], int)
            and isinstance(self.strategy_metadata[0][1], int)
        ):
            raise MisconfigurationException(
                "`unfreeze_milestones` strategy only accepts the format Tuple[Tuple[int, int], int]. HINT example: "
                "((5, 10), 15)."
            )

    def _get_modules_to_freeze(self, pl_module: LightningModule) -> Union[Module, Iterable[Union[Module, Iterable]]]:
        modules_to_freeze = getattr(pl_module, "modules_to_freeze", None)
        if modules_to_freeze is None:
            raise AttributeError(
                "LightningModule missing instance method 'modules_to_freeze'."
                "Please, implement the method which returns NoneType or a Module or an Iterable of Modules."
            )
        return modules_to_freeze()

    def freeze_before_training(self, pl_module: Union[Module, Iterable[Union[Module, Iterable]]]) -> None:
        if self.strategy != FinetuningStrategies.NO_FREEZE:
            modules = self._get_modules_to_freeze(pl_module=pl_module)
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

        modules = self._get_modules_to_freeze(pl_module=pl_module)
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

        modules = self._get_modules_to_freeze(pl_module=pl_module)
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


# Used for properly verifying input and providing neat and helpful error messages for users.
_DEFAULTS_FINETUNE_STRATEGIES = [
    "no_freeze",
    "freeze",
    "freeze_unfreeze",
    "unfreeze_milestones",
]

_FINETUNING_STRATEGIES_REGISTRY = FlashRegistry("finetuning_strategies")

if _PL_AVAILABLE:
    for strategy in FinetuningStrategies:
        _FINETUNING_STRATEGIES_REGISTRY(
            name=strategy.value,
            fn=partial(FlashBaseFinetuning, strategy_key=strategy),
        )


class NoFreeze(FlashBaseFinetuning):
    def __init__(self, train_bn: bool = True):
        super().__init__(FinetuningStrategies.NO_FREEZE, train_bn)


class Freeze(FlashBaseFinetuning):
    def __init__(self, train_bn: bool = True):
        super().__init__(FinetuningStrategies.FREEZE, train_bn)


class FreezeUnfreeze(FlashBaseFinetuning):
    def __init__(
        self,
        strategy_metadata: int,
        train_bn: bool = True,
    ):
        super().__init__(FinetuningStrategies.FREEZE_UNFREEZE, strategy_metadata, train_bn)


class UnfreezeMilestones(FlashBaseFinetuning):
    def __init__(
        self,
        strategy_metadata: Tuple[Tuple[int, int], int],
        train_bn: bool = True,
    ):
        super().__init__(FinetuningStrategies.UNFREEZE_MILESTONES, strategy_metadata, train_bn)
