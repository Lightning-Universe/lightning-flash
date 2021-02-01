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
        finetune_strategy: Optional[Union[str, Callback]] = None,
        **callbacks_kwargs,
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

            finetune_strategy: Should either be a string or a finetuning callback subclassing
                ``pytorch_lightning.callbacks.BaseFinetuning``.

            callbacks_kwargs: Those arguments will be provided to `model.configure_finetune_callbacks`
                to instantiante your own finetuning callbacks.

        """
        if isinstance(finetune_strategy, Callback) and not isinstance(finetune_strategy, BaseFinetuning):
            raise Exception("finetune_strategy should be a ``pytorch_lightning.callbacks.BaseFinetuning`` Callback")

        self._resolve_callbacks(model, finetune_strategy, **callbacks_kwargs)
        return super().fit(model, train_dataloader, val_dataloaders, datamodule)

    def _resolve_callbacks(self, model, finetune_strategy, **callbacks_kwargs):
        if sum((isinstance(c, BaseFinetuning) for c in [finetune_strategy])) > 1:
            raise MisconfigurationException("Only 1 callback subclassing `BaseFinetuning` should be provided.")
        # provided callbacks are higher priorities than model callbacks.
        callbacks = self.callbacks
        if isinstance(finetune_strategy, str):
            callbacks_kwargs["finetune_strategy"] = finetune_strategy
        else:
            callbacks = self._merge_callbacks(callbacks, [finetune_strategy])
        self.callbacks = self._merge_callbacks(callbacks, model.configure_finetune_callbacks(**callbacks_kwargs))

    @staticmethod
    def _merge_callbacks(current_callbacks: List, new_callbacks: List) -> List:
        if len(new_callbacks):
            return current_callbacks
        new_callbacks_types = set(type(c) for c in new_callbacks)
        current_callbacks_types = set(type(c) for c in current_callbacks)
        override_types = new_callbacks_types.intersection(current_callbacks_types)
        new_callbacks.extend(c for c in current_callbacks if type(c) not in override_types)
        return new_callbacks
