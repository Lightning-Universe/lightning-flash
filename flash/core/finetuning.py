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
from typing import Iterable, List, Optional, Tuple, Union

from pytorch_lightning.callbacks import BaseFinetuning
from torch.nn import Module, ModuleDict
from torch.optim import Optimizer

# Handle None case and just take one module using a single hook.

# Take a function that return the backbone(s).
# Use it again and again.


class FlashBaseFinetuning(BaseFinetuning):
    """FlashBaseFinetuning can be used to create a custom Flash Finetuning Callback."""

    def __init__(
        self,
        strategy_key: str,
        strategy_metadata: Optional[Union[int, Tuple[int, int]]],
        modules: Union[Module, Iterable[Union[Module, Iterable]]],
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

        self.strategy: str = strategy_key
        self.strategy_metadata: Optional[Union[int, Tuple[int, int]]] = strategy_metadata

        self.attr_names: List[str] = FlashBaseFinetuning.get_module_names(modules=modules)

        self.train_bn: bool = train_bn

    @staticmethod
    def get_module_names(modules: Union[Module, Iterable[Union[Module, Iterable]]]) -> List[str]:
        """This function is used to flatten a module or an iterable of modules into a list of its leaf modules
        (modules with no children) and parent modules that have parameters directly themselves and extract the
        names.

        Args:
            modules: A given module or an iterable of modules

        Returns:
            List of module names.
        """
        if modules is None:
            return []

        _module_names: List[str] = []
        _modules: Iterable[Module] = []
        if isinstance(modules, ModuleDict):
            _modules.extend(modules.values())
        elif isinstance(modules, Iterable):
            for module in modules:
                _module_names.extend(FlashBaseFinetuning.get_module_names(module))
        else:
            _modules.append(modules)
        for module in _modules:
            for name, _module in module.named_modules():
                if not list(_module.children()) or _module._parameters:
                    _module_names.append(name)
        return _module_names

    @staticmethod
    def _get_modules_from_attr_names(
        pl_module: Union[Module, Iterable[Union[Module, Iterable]]],
        attr_names: List[str],
    ) -> List[Union[Module, Iterable[Union[Module, Iterable]]]]:

        modules: List[Union[Module, Iterable[Union[Module, Iterable]]]] = []
        for attr_name in attr_names:
            _sub_module = pl_module.get_submodule(attr_name)
            # if not attr or not isinstance(attr, nn.Module):
            #     MisconfigurationException(f"Your model must have a {attr} attribute")
            modules.append(_sub_module)
        return modules

    def _freeze_using_attr_names(
        self, pl_module: Union[Module, Iterable[Union[Module, Iterable]]], train_bn: bool = True
    ) -> None:
        modules = FlashBaseFinetuning._get_modules_from_attr_names(pl_module=pl_module)
        self.freeze(modules=modules, train_bn=train_bn)

    def freeze_before_training(self, pl_module: Union[Module, Iterable[Union[Module, Iterable]]]) -> None:
        if self.strategy != "no_freeze":
            self._freeze_using_attr_names(pl_module, train_bn=self.train_bn)

    def _freeze_unfreeze_function(
        self,
        pl_module: Union[Module, Iterable[Union[Module, Iterable]]],
        epoch: int,
        optimizer: Optimizer,
        opt_idx: int,
        strategy_metadata: int,
    ):
        unfreeze_epoch: int = strategy_metadata
        # Common Implementation
        if epoch != unfreeze_epoch:
            return
        modules = FlashBaseFinetuning._get_modules_from_attr_names(pl_module=pl_module, attr_names=self.attr_names)
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

        # backbone_modules_attrs = list(pl_module.backbone.modules())
        if epoch == unfreeze_milestones[0]:
            # unfreeze num_layers last layers
            backbone_modules = FlashBaseFinetuning._get_modules_from_attr_names(
                pl_module=pl_module,
                attr_names=self.attr_names[-num_layers:],
            )
            self.unfreeze_and_add_param_group(
                modules=backbone_modules,
                optimizer=optimizer,
                train_bn=self.train_bn,
            )

        elif epoch == unfreeze_milestones[1]:
            # unfreeze remaining layers
            backbone_modules = FlashBaseFinetuning._get_modules_from_attr_names(
                pl_module=pl_module,
                attr_names=self.attr_names[:-num_layers],
            )
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
        if self.strategy in "freeze_unfreeze":
            self._freeze_unfreeze_function(pl_module, epoch, optimizer, opt_idx, self.strategy_metadata)
        elif self.strategy == "unfreeze_milestones":
            self._unfreeze_milestones_function(pl_module, epoch, optimizer, opt_idx, self.strategy_metadata)
        else:
            pass


_DEFAULTS_FINETUNE_STRATEGIES = [
    "no_freeze",
    "freeze",
    "freeze_unfreeze",
    "unfreeze_milestones",
]
