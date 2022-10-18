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
import os
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import BaseFinetuning
from torch.nn import Module
from torch.optim import Optimizer

from flash.core.registry import FlashRegistry

if not os.environ.get("READTHEDOCS", False):
    from pytorch_lightning.utilities.enums import LightningEnum
else:
    # ReadTheDocs mocks the `LightningEnum` import to be a regular type, so we replace it with a plain Enum here.
    from enum import Enum

    LightningEnum = Enum


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
        strategy_key: Union[str, FinetuningStrategies],
        strategy_metadata: Optional[Union[int, Tuple[Tuple[int, int], int]]] = None,
        train_bn: bool = True,
    ):
        """
        Args:
            strategy_key: The finetuning strategy to be used. See :meth:`~flash.core.trainer.Trainer.finetune`
                for the available strategies.
            strategy_metadata: Data that accompanies certain finetuning strategies like epoch number or number of
                layers.
            train_bn: Whether to train Batch Norm layer
        """
        super().__init__()

        self.strategy: FinetuningStrategies = strategy_key
        self.strategy_metadata: Optional[Union[int, Tuple[Tuple[int, int], int]]] = strategy_metadata
        self.train_bn: bool = train_bn

        if self.strategy == FinetuningStrategies.FREEZE_UNFREEZE and not isinstance(self.strategy_metadata, int):
            raise TypeError(
                "The `freeze_unfreeze` strategy requires an integer denoting the epoch number to unfreeze at. Example: "
                "`strategy=('freeze_unfreeze', 7)`"
            )
        if self.strategy == FinetuningStrategies.UNFREEZE_MILESTONES and not (
            isinstance(self.strategy_metadata, Tuple)
            and isinstance(self.strategy_metadata[0], Tuple)
            and isinstance(self.strategy_metadata[1], int)
            and isinstance(self.strategy_metadata[0][0], int)
            and isinstance(self.strategy_metadata[0][1], int)
        ):
            raise TypeError(
                "The `unfreeze_milestones` strategy requires the format Tuple[Tuple[int, int], int]. Example: "
                "`strategy=('unfreeze_milestones', ((5, 10), 15))`"
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

    def unfreeze_and_extend_param_group(
        self,
        modules: Union[Module, Iterable[Union[Module, Iterable]]],
        optimizer: Optimizer,
        train_bn: bool = True,
    ) -> None:
        self.make_trainable(modules)

        params = self.filter_params(modules, train_bn=train_bn, requires_grad=True)
        params = self.filter_on_optimizer(optimizer, params)
        if params:
            optimizer.param_groups[0]["params"].extend(params)

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
            self.unfreeze_and_extend_param_group(
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
                self.unfreeze_and_extend_param_group(
                    modules=backbone_modules,
                    optimizer=optimizer,
                    train_bn=self.train_bn,
                )
            elif epoch == unfreeze_milestones[1]:
                # unfreeze remaining layers
                backbone_modules = BaseFinetuning.flatten_modules(modules=modules)[:-num_layers]
                self.unfreeze_and_extend_param_group(
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


_FINETUNING_STRATEGIES_REGISTRY = FlashRegistry("finetuning_strategies")

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


class FlashDeepSpeedFinetuning(FlashBaseFinetuning):
    """FlashDeepSpeedFinetuning can be used to create a custom Flash Finetuning Callback which works with
    DeepSpeed.

    DeepSpeed cannot store and load its parameters when working with Lightning. So FlashDeepSpeedFinetuning overrides
    `_store` to not store its parameters.
    """

    def _store(
        self,
        pl_module: LightningModule,
        opt_idx: int,
        num_param_groups: int,
        current_param_groups: List[Dict[str, Any]],
    ) -> None:
        pass


class NoFreezeDeepSpeed(FlashDeepSpeedFinetuning):
    def __init__(self, train_bn: bool = True):
        super().__init__(FinetuningStrategies.NO_FREEZE, train_bn)


class FreezeDeepSpeed(FlashDeepSpeedFinetuning):
    def __init__(self, train_bn: bool = True):
        super().__init__(FinetuningStrategies.FREEZE, train_bn)


class FreezeUnfreezeDeepSpeed(FlashDeepSpeedFinetuning):
    def __init__(
        self,
        strategy_metadata: int,
        train_bn: bool = True,
    ):
        super().__init__(FinetuningStrategies.FREEZE_UNFREEZE, strategy_metadata, train_bn)


class UnfreezeMilestonesDeepSpeed(FlashDeepSpeedFinetuning):
    def __init__(
        self,
        strategy_metadata: Tuple[Tuple[int, int], int],
        train_bn: bool = True,
    ):
        super().__init__(FinetuningStrategies.UNFREEZE_MILESTONES, strategy_metadata, train_bn)


for strategy in FinetuningStrategies:
    _FINETUNING_STRATEGIES_REGISTRY(
        name=f"{strategy.value}_deepspeed",
        fn=partial(FlashDeepSpeedFinetuning, strategy_key=strategy),
    )
