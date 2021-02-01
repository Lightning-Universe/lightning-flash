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
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.optim import Optimizer

_EXCLUDE_PARAMTERS = ("self", "args", "kwargs")


class NeverFreeze(BaseFinetuning):
    pass


class NeverUnfreeze(BaseFinetuning):

    def __init__(self, attr_names: Union[str, List[str]] = "backbone", train_bn: bool = True):
        self.attr_names = [attr_names] if isinstance(attr_names, str) else attr_names
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module: pl.LightningModule) -> None:
        for attr_name in self.attr_names:
            attr = getattr(pl_module, attr_name, None)
            if attr is None or not isinstance(attr, nn.Module):
                MisconfigurationException("To use NeverUnfreeze your model must have a {attr} attribute")
            self.freeze(module=attr, train_bn=self.train_bn)


class FreezeUnFreeze(NeverUnfreeze):

    def __init__(
        self, attr_names: Union[str, List[str]] = "backbone", train_bn: bool = True, unfreeze_at_epoch: int = 10
    ):
        super().__init__(attr_names, train_bn)
        self.unfreeze_at_epoch = unfreeze_at_epoch

    def finetunning_function(
        self,
        pl_module: pl.LightningModule,
        epoch: int,
        optimizer: Optimizer,
        opt_idx: int,
    ) -> None:
        if epoch == self.unfreeze_at_epoch:
            modules = []
            for attr_name in self.attr_names:
                modules.append(getattr(pl_module, attr_name))

            self.unfreeze_and_add_param_group(
                module=modules,
                optimizer=optimizer,
                train_bn=self.train_bn,
            )


class MilestonesFinetuning(BaseFinetuning):

    def __init__(self, unfreeze_milestones: tuple = (5, 10), train_bn: bool = True, num_layers: int = 5):
        self.unfreeze_milestones = unfreeze_milestones
        self.train_bn = train_bn
        self.num_layers = num_layers

    def freeze_before_training(self, pl_module: pl.LightningModule) -> None:
        # TODO: might need some config to say which attribute is model
        # maybe something like:
        # self.freeze(module=pl_module.getattr(self.feature_attr), train_bn=self.train_bn)
        # where self.feature_attr can be "backbone" or "feature_extractor", etc.
        # (configured in init)
        assert hasattr(pl_module, "backbone"), "To use MilestonesFinetuning your model must have a backbone attribute"
        self.freeze(module=pl_module.backbone, train_bn=self.train_bn)

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


def instantiate_cls(cls, kwargs):
    parameters = list(inspect.signature(cls.__init__).parameters.keys())
    parameters = [p for p in parameters if p not in _EXCLUDE_PARAMTERS]
    cls_kwargs = {}
    for p in parameters:
        if p in kwargs:
            cls_kwargs[p] = kwargs.pop(p)
    if len(kwargs) > 0:
        raise MisconfigurationException(f"Available parameters are: {parameters}. Found {kwargs} left")
    return cls(**cls_kwargs)


_DEFAULTS_FINETUNE_STRATEGIES = {
    "never_freeze": NeverFreeze,
    "never_unfreeze": NeverUnfreeze,
    "freeze_unfreeze": FreezeUnFreeze,
    "unfreeze_milestones": MilestonesFinetuning
}


def instantiate_default_finetuning_callbacks(kwargs):
    finetune_strategy = kwargs.pop("finetune_strategy", None)
    if isinstance(finetune_strategy, str):
        finetune_strategy = finetune_strategy.lower()
        if finetune_strategy in _DEFAULTS_FINETUNE_STRATEGIES:
            return [instantiate_cls(_DEFAULTS_FINETUNE_STRATEGIES[finetune_strategy], kwargs)]
        else:
            msg = "\n Extra arguments can be: \n"
            for n, cls in _DEFAULTS_FINETUNE_STRATEGIES.items():
                parameters = list(inspect.signature(cls.__init__).parameters.keys())
                parameters = [p for p in parameters if p not in _EXCLUDE_PARAMTERS]
                msg += f"{n}: {parameters} \n"
            raise MisconfigurationException(
                f"finetune_strategy should be within {list(_DEFAULTS_FINETUNE_STRATEGIES)}"
                f"{msg}"
                f". Found {finetune_strategy}"
            )
    return []
