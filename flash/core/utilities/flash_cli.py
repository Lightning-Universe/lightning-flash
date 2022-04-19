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
from argparse import Namespace
from functools import wraps
from inspect import Parameter, signature
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

import pytorch_lightning as pl
from jsonargparse import ArgumentParser
from jsonargparse.signatures import get_class_signature_functions
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities.model_helpers import is_overridden

import flash
from flash.core.data.data_module import DataModule
from flash.core.utilities.lightning_cli import (
    class_from_function,
    LightningArgumentParser,
    LightningCLI,
    SaveConfigCallback,
)
from flash.core.utilities.stability import beta


class ModelExcludeSaveConfigCallback(SaveConfigCallback):
    """An override of the ``SaveConfigCallback`` that excludes chosen parameters from the model config."""

    def __init__(
        self,
        parser: LightningArgumentParser,
        config: Union[Namespace, Dict[str, Any]],
        config_filename: str,
        overwrite: bool = False,
        exclude_list: Optional[List[str]] = None,
    ) -> None:
        if exclude_list is None:
            exclude_list = []
        self.exclude_list = exclude_list

        super().__init__(parser, config, config_filename, overwrite)

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        for parameter in self.exclude_list:
            del self.config["model"][parameter]
        return super().setup(trainer, pl_module, stage)


def drop_kwargs(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    # Override signature
    sig = signature(func)
    sig = sig.replace(
        parameters=tuple(p for p in sig.parameters.values() if p.kind is not p.VAR_KEYWORD and p.name != "self")
    )
    if inspect.isclass(func):
        sig = sig.replace(return_annotation=func)
    wrapper.__signature__ = sig

    return wrapper


def get_kwarg_name(func) -> Optional[str]:
    sig = signature(func)
    var_kwargs = [p for p in sig.parameters.values() if p.kind == p.VAR_KEYWORD]
    if len(var_kwargs) == 1:
        return var_kwargs[0].name
    return None


def make_args_optional(cls, args: Set[str]):
    @wraps(cls)
    def wrapper(*args, **kwargs):
        return cls(*args, **kwargs)

    # Override signature
    sig = signature(cls)
    parameters = [p for p in sig.parameters.values() if p.name not in args or p.default != p.empty]
    filtered_parameters = [p for p in sig.parameters.values() if p.name in args and p.default == p.empty]

    index = [i for i, p in enumerate(parameters) if p.kind == p.VAR_KEYWORD]
    index = len(parameters) if not index else index[0]

    for p in filtered_parameters:
        new_parameter = Parameter(p.name, p.POSITIONAL_OR_KEYWORD, default=None, annotation=Optional[p.annotation])
        parameters.insert(index, new_parameter)

    sig = sig.replace(parameters=parameters, return_annotation=cls)
    wrapper.__signature__ = sig

    return wrapper


def get_overlapping_args(func_a, func_b) -> Set[str]:
    func_a = get_class_signature_functions([func_a])[0][1]
    func_b = get_class_signature_functions([func_b])[0][1]
    return set(inspect.signature(func_a).parameters.keys() & inspect.signature(func_b).parameters.keys())


@beta("Flash Zero is currently in Beta.")
class FlashCLI(LightningCLI):

    datamodule: DataModule
    config_init: Namespace
    model: LightningModule

    def __init__(
        self,
        model_class: Type[pl.LightningModule],
        datamodule_class: Type["flash.DataModule"],
        trainer_class: Type[pl.Trainer] = flash.Trainer,
        default_datamodule_builder: Optional[Callable] = None,
        additional_datamodule_builders: Optional[List[Callable]] = None,
        default_arguments=None,
        finetune=True,
        datamodule_attributes=None,
        **kwargs: Any,
    ) -> None:
        """Flash's extension of the :class:`pytorch_lightning.utilities.cli.LightningCLI`

        Args:
            model_class: The :class:`pytorch_lightning.LightningModule` class to train on.
            datamodule_class: The :class:`~flash.data.data_module.DataModule` class.
            trainer_class: An optional extension of the :class:`pytorch_lightning.Trainer` class.
            trainer_fn: The trainer function to run.
            input: Use this if your ``DataModule`` is created using a classmethod. Any of:
                - ``None``. The ``datamodule_class.__init__`` signature will be used.
                - ``str``. One of :class:`~flash.data.io.input.InputFormat`. This will use the signature of
                    the corresponding ``DataModule.from_*`` method.
                - ``Callable``. A custom method.
            kwargs: See the parent arguments
        """
        if datamodule_attributes is None:
            datamodule_attributes = {"num_classes"}
        self.datamodule_attributes = datamodule_attributes

        self.default_datamodule_builder = default_datamodule_builder
        self.additional_datamodule_builders = additional_datamodule_builders or []
        self.default_arguments = default_arguments or {}
        self.finetune = finetune

        model_class = make_args_optional(model_class, self.datamodule_attributes)
        self.local_datamodule_class = datamodule_class

        self._subcommand_builders = {}

        super().__init__(
            drop_kwargs(model_class),
            datamodule_class=None,
            save_config_callback=functools.partial(ModelExcludeSaveConfigCallback, exclude_list=datamodule_attributes),
            trainer_class=trainer_class,
            **kwargs,
        )

    @contextlib.contextmanager
    def patch_default_subcommand(self):
        parse_common = self.parser._parse_common

        if self.default_datamodule_builder is not None:

            @functools.wraps(parse_common)
            def wrapper(cfg, *args, **kwargs):
                if "subcommand" not in cfg or cfg["subcommand"] is None:
                    cfg["subcommand"] = self.default_datamodule_builder.__name__
                return parse_common(cfg, *args, **kwargs)

            self.parser._parse_common = wrapper

        yield

        self.parser._parse_common = parse_common

    def parse_arguments(self) -> None:
        with self.patch_default_subcommand():
            super().parse_arguments()

    def add_arguments_to_parser(self, parser) -> None:
        subcommands = parser.add_subcommands()

        for function in vars(self.local_datamodule_class).keys():
            if not function.startswith("from"):
                continue
            if (
                hasattr(DataModule, function) and is_overridden(function, self.local_datamodule_class, DataModule)
            ) or not hasattr(DataModule, function):
                if getattr(self.local_datamodule_class, function, None) is not None:
                    self.add_subcommand_from_function(subcommands, getattr(self.local_datamodule_class, function))

        for datamodule_builder in self.additional_datamodule_builders:
            self.add_subcommand_from_function(subcommands, datamodule_builder)

        if self.default_datamodule_builder is not None:
            self.add_subcommand_from_function(subcommands, self.default_datamodule_builder)

        parser.set_defaults(self.default_arguments)

    def add_subcommand_from_function(self, subcommands, function, function_name=None):
        subcommand = ArgumentParser()
        if get_kwarg_name(function) == "data_module_kwargs":
            datamodule_function = class_from_function(function, return_type=self.local_datamodule_class)
            subcommand.add_class_arguments(
                datamodule_function,
                fail_untyped=False,
                skip={
                    "self",
                    "train_dataset",
                    "val_dataset",
                    "test_dataset",
                    "predict_dataset",
                    "train_input",
                    "val_input",
                    "test_input",
                    "predict_input",
                    "input",
                    "input_transform",
                },
            )
        else:
            datamodule_function = class_from_function(drop_kwargs(function), return_type=self.local_datamodule_class)
            subcommand.add_class_arguments(datamodule_function, fail_untyped=False)
        subcommand_name = function_name or function.__name__
        subcommands.add_subcommand(subcommand_name, subcommand)
        self._subcommand_builders[subcommand_name] = function

    def instantiate_classes(self) -> None:
        """Instantiates the classes using settings from self.config."""
        sub_config = self.config.get("subcommand")
        self.datamodule = self._subcommand_builders[sub_config](**self.config.get(sub_config))

        for datamodule_attribute in self.datamodule_attributes:
            if datamodule_attribute in self.config["model"]:
                if getattr(self.datamodule, datamodule_attribute, None) is not None:
                    self.config["model"][datamodule_attribute] = getattr(self.datamodule, datamodule_attribute)
        self.config_init = self.parser.instantiate_classes(self.config)
        self.model = self.config_init["model"]
        self.instantiate_trainer()

    def prepare_fit_kwargs(self):
        super().prepare_fit_kwargs()
        if self.finetune:
            # TODO: expose the strategy arguments?
            self.fit_kwargs["strategy"] = "freeze"

    def fit(self) -> None:
        if self.finetune:
            self.trainer.finetune(**self.fit_kwargs)
        else:
            self.trainer.fit(**self.fit_kwargs)
