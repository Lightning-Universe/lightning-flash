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
import warnings
from typing import List, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import BaseFinetuning, Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader

from flash.core.finetuning import instantiate_default_finetuning_callbacks
from flash.core.model import Task


class Trainer(pl.Trainer):

    def fit(
        self,
        model: pl.LightningModule,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        datamodule: Optional[pl.LightningDataModule] = None,
    ):
        r"""
        Runs the full optimization routine. Same as pytorch_lightning.Trainer().fit()

        Args:
            datamodule: A instance of :class:`LightningDataModule`.

            model: Model to fit.

            train_dataloader: A Pytorch DataLoader with training samples. If the model has
                a predefined train_dataloader method this will be skipped.

            val_dataloaders: Either a single Pytorch Dataloader or a list of them, specifying validation samples.
                If the model has a predefined val_dataloaders method this will be skipped
        """
        if any(isinstance(c, BaseFinetuning) for c in self.callbacks):
            # TODO: if we find a finetuning callback in the trainer should we remove it? or just warn the user?
            warnings.warn("Warning: You are calling fit(), but your trainer is using a fine-tuning callback")
        return super().fit(model, train_dataloader, val_dataloaders, datamodule)

    def finetune(
        self,
        model: Task,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        datamodule: Optional[pl.LightningDataModule] = None,
        strategy: Optional[Union[str, BaseFinetuning]] = None,
    ):
        r"""
        Runs the full optimization routine. Same as pytorch_lightning.Trainer().fit(), but unfreezes layers
        of the backbone throughout training layers of the backbone throughout training.

        Args:
            datamodule: A instance of :class:`LightningDataModule`.

            model: Model to fit.

            train_dataloader: A Pytorch DataLoader with training samples. If the model has
                a predefined train_dataloader method this will be skipped.

            val_dataloaders: Either a single Pytorch Dataloader or a list of them, specifying validation samples.
                If the model has a predefined val_dataloaders method this will be skipped

            strategy: Should either be a string or a finetuning callback subclassing
                ``pytorch_lightning.callbacks.BaseFinetuning``.
                Currently default strategies can be create with strings such as:
                    * ``no_freeze``,
                    * ``freeze``
                    * ``freeze_unfreeze``
                    * ``unfreeze_milestones``

        """
        if not isinstance(strategy, (BaseFinetuning, str)):
            raise MisconfigurationException(
                "strategy should be a ``pytorch_lightning.callbacks.BaseFinetuning`` Callback or a str"
            )

        self._resolve_callbacks(strategy)
        return super().fit(model, train_dataloader, val_dataloaders, datamodule)

    def _resolve_callbacks(self, strategy):
        if sum((isinstance(c, BaseFinetuning) for c in [strategy])) > 1:
            raise MisconfigurationException("Only 1 callback subclassing `BaseFinetuning` should be provided.")
        # todo: change to ``configure_callbacks`` when merged to Lightning.
        callbacks = self.callbacks
        if isinstance(strategy, str):
            strategy = instantiate_default_finetuning_callbacks(strategy)
        self.callbacks = self._merge_callbacks(callbacks, [strategy])

    @staticmethod
    def _merge_callbacks(current_callbacks: List, new_callbacks: List) -> List:
        if len(new_callbacks):
            return current_callbacks
        new_callbacks_types = set(type(c) for c in new_callbacks)
        current_callbacks_types = set(type(c) for c in current_callbacks)
        override_types = new_callbacks_types.intersection(current_callbacks_types)
        new_callbacks.extend(c for c in current_callbacks if type(c) not in override_types)
        return new_callbacks
