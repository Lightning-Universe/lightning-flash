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
import warnings
from argparse import ArgumentParser, Namespace
from functools import wraps
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning import Trainer as PlTrainer
from pytorch_lightning.callbacks import BaseFinetuning
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.utilities.argparse import add_argparse_args, get_init_arguments_and_types, parse_env_variables
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from torch.utils.data import DataLoader

import flash
from flash.core.model import Task
from flash.core.utilities.imports import _PL_GREATER_EQUAL_1_5_0, _SERVE_AVAILABLE


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
    def __init__(self, *args, serve_sanity_check: bool = False, **kwargs):
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
        super().__init__(*args, **kwargs)

        self.serve_sanity_check = serve_sanity_check

    def _run_sanity_check(self, ref_model):
        if hasattr(super(), "_run_sanity_check"):
            super()._run_sanity_check(ref_model)

        self.run_sanity_check(ref_model)

    def run_sanity_check(self, ref_model):
        if hasattr(super(), "run_sanity_check"):
            super().run_sanity_check(ref_model)

        if self.serve_sanity_check and ref_model.is_servable and _SERVE_AVAILABLE:
            ref_model.run_serve_sanity_check()

    # TODO @(tchaton) remove `reset_train_val_dataloaders` from run_train function
    def _run_train(self) -> None:
        self._pre_training_routine()

        if not self.is_global_zero and self.progress_bar_callback is not None:
            self.progress_bar_callback.disable()

        self._run_sanity_check(self.lightning_module)

        # enable train mode
        self.model.train()
        torch.set_grad_enabled(True)

        # reload data when needed
        model = self.lightning_module

        if isinstance(self.fit_loop, FitLoop):
            self.reset_train_val_dataloaders(model)

        self.fit_loop.trainer = self
        self.fit_loop.run()

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
                - ``("freeze_unfreeze", integer)``
                - ``("unfreeze_milestones", ((integer, integer), integer))``

                where ``integer`` can be any integer.
                By default, "no_freeze" strategy will be used.

            train_bn: Whether to train Batch Norm layer
        """
        self._resolve_callbacks(model, strategy, train_bn=train_bn)
        return super().fit(model, train_dataloader, val_dataloaders, datamodule)

    def _resolve_callbacks(
        self,
        model: Task,
        strategy: Union[str, BaseFinetuning, Tuple[str, int], Tuple[str, Tuple[int, int]]] = "no_freeze",
        train_bn: bool = True,
    ):
        """This function is used to select the `BaseFinetuning` to be used for finetuning."""
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

    def _parse_request_dataloader_args(self, args: Tuple, kwargs: Dict):
        """Handles backwards compatibility for ``request_dataloader``.

        Possible combinations:

        legacy: (model, stage)
        (stage, model)
        (stage, model=model)
        """
        model, stage, is_legacy = None, None, False
        if len(args) == 2:
            # Check for legacy arguments: (model, stage)
            if isinstance(args[0], LightningModule):
                is_legacy = True
                model, stage = args
            else:  # (stage, model)
                stage, model = args
        else:
            stage = kwargs.get("stage", args[0])
            model = kwargs.get("model")
        return model, stage, is_legacy

    def request_dataloader(
        self,
        *args,
        **kwargs,
    ) -> Union[DataLoader, List[DataLoader]]:
        """Handles downloading data in the GPU or TPU case.

        Returns:
            The dataloader
        """
        model, stage, is_legacy = self._parse_request_dataloader_args(args, kwargs)

        if is_legacy:
            self.call_hook(f"on_{stage}_dataloader")
            dataloader = getattr(model, f"{stage}_dataloader")()
        else:
            hook = f"{stage.dataloader_prefix}_dataloader"
            self.call_hook("on_" + hook, pl_module=model)

            if is_overridden(hook, model):
                dataloader = self.call_hook(hook, pl_module=model)
            elif _PL_GREATER_EQUAL_1_5_0:
                source = getattr(self._data_connector, f"_{stage.dataloader_prefix}_dataloader_source")
                dataloader = source.dataloader()

        if isinstance(dataloader, tuple):
            dataloader = list(dataloader)
        if _PL_GREATER_EQUAL_1_5_0:
            self.training_type_plugin.barrier("get_dataloaders")
        else:
            self.accelerator.barrier("get_dataloaders")
        return dataloader
