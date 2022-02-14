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
import contextlib
import functools
import inspect
import warnings
from argparse import ArgumentParser, Namespace
from functools import wraps
from typing import Callable, List, Optional, Tuple, Union

import torch
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning import Trainer as PlTrainer
from pytorch_lightning.callbacks import BaseFinetuning
from pytorch_lightning.utilities.argparse import add_argparse_args, get_init_arguments_and_types, parse_env_variables
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader

import flash
from flash.core.data.io.output import Output
from flash.core.data.io.output_transform import OutputTransform
from flash.core.model import Task
from flash.core.registry import FlashRegistry


def from_argparse_args(cls, args: Union[Namespace, ArgumentParser], **kwargs):
    """Modified version of :func:`pytorch_lightning.utilities.argparse.from_argparse_args` which populates
    ``valid_kwargs`` from :class:`pytorch_lightning.Trainer`."""
    if isinstance(args, ArgumentParser):
        args = cls.parse_argparser(args)

    params = vars(args)

    # we only want to pass in valid PLTrainer args, the rest may be user specific
    valid_kwargs = inspect.signature(PlTrainer.__init__).parameters
    trainer_kwargs = {name: params[name] for name in valid_kwargs if name in params}
    trainer_kwargs.update(**kwargs)

    return cls(**trainer_kwargs)


def _defaults_from_env_vars(fn: Callable) -> Callable:
    """Copy of ``pytorch_lightning.trainer.connectors.env_vars_connector._defaults_from_env_vars``.

    Required to fix build error in readthedocs.
    """

    @wraps(fn)
    def insert_env_defaults(self, *args, **kwargs):
        cls = self.__class__  # get the class
        if args:  # inace any args passed move them to kwargs
            # parse only the argument names
            cls_arg_names = [arg[0] for arg in get_init_arguments_and_types(cls)]
            # convert args to kwargs
            kwargs.update(dict(zip(cls_arg_names, args)))
        env_variables = vars(parse_env_variables(cls))
        # update the kwargs by env variables
        kwargs = dict(list(env_variables.items()) + list(kwargs.items()))

        # all args were already moved to kwargs
        return fn(self, **kwargs)

    return insert_env_defaults


class Trainer(PlTrainer):
    @_defaults_from_env_vars
    def __init__(self, *args, **kwargs):
        if flash._IS_TESTING:
            if torch.cuda.is_available():
                kwargs["gpus"] = 1
                kwargs["max_epochs"] = 3
                kwargs["limit_train_batches"] = 1.0
                kwargs["limit_val_batches"] = 1.0
                kwargs["limit_test_batches"] = 1.0
                kwargs["fast_dev_run"] = False
            else:
                kwargs["fast_dev_run"] = True
                kwargs["gpus"] = None
                kwargs["accelerator"] = None
                kwargs["precision"] = 32
        super().__init__(*args, **kwargs)

    def fit(
        self,
        model: LightningModule,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        datamodule: Optional[LightningDataModule] = None,
    ):
        r"""
        Runs the full optimization routine. Same as :meth:`pytorch_lightning.Trainer.fit`

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
        model: LightningModule,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        datamodule: Optional[LightningDataModule] = None,
        strategy: Union[str, BaseFinetuning, Tuple[str, int], Tuple[str, Tuple[int, int]]] = "no_freeze",
        train_bn: bool = True,
    ):
        r"""

        Runs the full optimization routine. Same as :meth:`pytorch_lightning.Trainer.fit`, but unfreezes layers
        of the backbone throughout training layers of the backbone throughout training.

        Args:
            model: Model to fit.

            train_dataloader: A PyTorch DataLoader with training samples. If the model has
                a predefined train_dataloader method this will be skipped.

            val_dataloaders: Either a single PyTorch Dataloader or a list of them, specifying validation samples.
                If the model has a predefined val_dataloaders method this will be skipped

            datamodule: A instance of :class:`LightningDataModule`.

            strategy: Should either be a string, or a Tuple, or a finetuning callback subclassing
                :class:`pytorch_lightning.callbacks.BaseFinetuning`.

                Default strategies can be enabled with these inputs:

                - ``"no_freeze"``
                - ``"freeze"``
                - ``("freeze_unfreeze", integer: unfreeze_epoch)``
                - ``("unfreeze_milestones", ((integer: unfreeze_epoch_num_layers, integer: unfreeze_epoch_all_layers),
                  integer: num_layers))``

                where ``integer`` can be any integer.
                By default, ``no_freeze`` strategy will be used.

            train_bn: Whether to train Batch Norm layer
        """
        self._resolve_callbacks(model, strategy, train_bn=train_bn)
        return super().fit(model, train_dataloader, val_dataloaders, datamodule)

    @contextlib.contextmanager
    def _wrap_predict_step(self, model, output_transform, output) -> None:
        predict_step = model.predict_step

        @functools.wraps(predict_step)
        def wrapper(*args, **kwargs):
            predictions = predict_step(*args, **kwargs)
            if predictions is not None:
                predictions = output_transform(predictions)
                predictions = [output(prediction) for prediction in predictions]
            return predictions

        model.predict_step = wrapper
        try:
            yield
        finally:
            model.predict_step = predict_step

    def predict(
        self,
        model: Optional[LightningModule] = None,
        dataloaders: Optional[Union[DataLoader, LightningDataModule]] = None,
        output: Union[Output, str] = None,
        **kwargs,
    ):
        r"""
        Run inference on your data.
        This will call the model forward function to compute predictions. Useful to perform distributed
        and batched predictions. Logging is disabled in the predict hooks.

        Args:
            model: The model to predict with.
            dataloaders: A :class:`torch.utils.data.DataLoader` or a sequence of them,
                or a :class:`~pytorch_lightning.core.datamodule.LightningDataModule` specifying prediction samples.
            output: The :class:`~flash.core.data.io.output.Output` to use to transform predict outputs.
            kwargs: Additional keyword arguments to pass to ``pytorch_lightning.Trainer.predict``.


        Returns:
            Returns a list of dictionaries, one for each provided dataloader containing their respective predictions.
        """
        model = model or self.lightning_module
        output_transform = getattr(model, "_output_transform", None) or OutputTransform()
        if output is None:
            output = Output()
        if isinstance(output, str) and isinstance(model, Task):
            output = getattr(model, "outputs", FlashRegistry("outputs")).get(output).from_task(model)

        with self._wrap_predict_step(model, output_transform, output):
            return super().predict(model, dataloaders, **kwargs)

    def _resolve_callbacks(
        self,
        model: Task,
        strategy: Union[str, BaseFinetuning, Tuple[str, int], Tuple[str, Tuple[int, int]]] = "no_freeze",
        train_bn: bool = True,
    ):
        """This function is used to select the `BaseFinetuning` to be used for finetuning."""
        if isinstance(strategy, str) and strategy == "no_freeze":
            warnings.warn("The model contains a default finetune callback.", UserWarning)
        finetuning_callback = model.configure_finetune_callback(strategy=strategy, train_bn=train_bn)
        if len(finetuning_callback) > 1:
            raise MisconfigurationException("Create a list with only 1 finetuning callback.")
        self.callbacks = self._merge_callbacks(self.callbacks, finetuning_callback)

    @staticmethod
    def _merge_callbacks(old_callbacks: List, new_callbacks: List) -> List:
        """This function keeps only 1 instance of each callback type, extending new_callbacks with
        old_callbacks."""
        if len(new_callbacks) == 0:
            return old_callbacks
        new_callbacks_types = {type(c) for c in new_callbacks}
        old_callbacks_types = {type(c) for c in old_callbacks}
        override_types = new_callbacks_types.intersection(old_callbacks_types)
        new_callbacks.extend(c for c in old_callbacks if type(c) not in override_types)
        return new_callbacks

    @classmethod
    def add_argparse_args(cls, *args, **kwargs) -> ArgumentParser:
        """See :func:`pytorch_lightning.utilities.argparse.add_argparse_args`."""
        # the lightning trainer implementation does not support subclasses.
        # context: https://github.com/PyTorchLightning/lightning-flash/issues/342#issuecomment-848892447
        return add_argparse_args(PlTrainer, *args, **kwargs)

    @classmethod
    def from_argparse_args(cls, args: Union[Namespace, ArgumentParser], **kwargs) -> "Trainer":
        """Modified version of :func:`pytorch_lightning.utilities.argparse.from_argparse_args` which populates
        ``valid_kwargs`` from :class:`pytorch_lightning.Trainer`."""
        # the lightning trainer implementation does not support subclasses.
        # context: https://github.com/PyTorchLightning/lightning-flash/issues/342#issuecomment-848892447
        return from_argparse_args(Trainer, args, **kwargs)
