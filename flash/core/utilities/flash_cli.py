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
from typing import Any, Callable, Dict, Optional, Set, Type, Union

import pytorch_lightning as pl

import flash
from flash.core.utilities.lightning_cli import LightningCLI


class FlashCLI(LightningCLI):
    def __init__(
        self,
        model_class: Optional[Union[Type[pl.LightningModule], Callable[..., pl.LightningModule]]] = None,
        datamodule_class: Optional[Union[Type[flash.DataModule], Callable[..., flash.DataModule]]] = None,
        trainer_class: Union[Type[flash.Trainer], Callable[..., flash.Trainer]] = flash.Trainer,
        **kwargs: Any,
    ) -> None:
        """Flash's extension of the :class:`pytorch_lightning.utilities.cli.LightningCLI`

        Args:
            model_class: The :class:`pytorch_lightning.LightningModule` class to train on.
            datamodule_class: The :class:`~flash.core.data.data_module.DataModule` class.
            trainer_class: An optional extension of the :class:`~flash.core.trainer.Trainer` class.
            **kwargs: See the parent arguments.
        """
        super().__init__(
            model_class=model_class, datamodule_class=datamodule_class, trainer_class=trainer_class, **kwargs
        )

    @staticmethod
    def subcommands() -> Dict[str, Set[str]]:
        subcommands = LightningCLI.subcommands()
        subcommands["finetune"] = subcommands["fit"]
        return subcommands
