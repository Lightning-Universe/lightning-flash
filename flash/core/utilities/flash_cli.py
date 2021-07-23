import contextlib
import functools
from functools import wraps
from inspect import Parameter, signature
from typing import Any, Optional, Set, Type

import pytorch_lightning as pl
from jsonargparse import ArgumentParser

import flash
from flash.core.utilities.lightning_cli import class_from_function, LightningCLI


def drop_kwargs(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    # Override signature
    sig = signature(func)
    sig = sig.replace(parameters=tuple(p for p in sig.parameters.values() if p.kind is not p.VAR_KEYWORD))
    wrapper.__signature__ = sig

    return wrapper


def make_args_optional(cls, args: Set[str]):

    @wraps(cls)
    def wrapper(*args, **kwargs):
        return cls(*args, **kwargs)

    # Override signature
    sig = signature(cls)
    parameters = [p for p in sig.parameters.values() if p.name not in args]
    filtered_parameters = [p for p in sig.parameters.values() if p.name in args]

    index = [i for i, p in enumerate(parameters) if p.kind == p.VAR_KEYWORD]
    if index == []:
        index = len(parameters)
    else:
        index = index[0]

    for p in filtered_parameters:
        new_parameter = Parameter(p.name, p.POSITIONAL_OR_KEYWORD, default=None, annotation=Optional[p.annotation])
        parameters.insert(index, new_parameter)

    sig = sig.replace(parameters=parameters, return_annotation=cls)
    wrapper.__signature__ = sig

    return wrapper


class FlashCLI(LightningCLI):

    def __init__(
        self,
        model_class: Type[pl.LightningModule],
        datamodule_class: Type['flash.DataModule'],
        trainer_class: Type[pl.Trainer] = flash.Trainer,
        default_subcommand="from_folders",
        default_arguments=None,
        finetune=True,
        **kwargs: Any,
    ) -> None:
        """
        Flash's extension of the :class:`pytorch_lightning.utilities.cli.LightningCLI`

        Args:
            model_class: The :class:`pytorch_lightning.LightningModule` class to train on.
            datamodule_class: The :class:`~flash.data.data_module.DataModule` class.
            trainer_class: An optional extension of the :class:`pytorch_lightning.Trainer` class.
            trainer_fn: The trainer function to run.
            datasource: Use this if your ``DataModule`` is created using a classmethod. Any of:
                - ``None``. The ``datamodule_class.__init__`` signature will be used.
                - ``str``. One of :class:`~flash.data.data_source.DefaultDataSources`. This will use the signature of
                    the corresponding ``DataModule.from_*`` method.
                - ``Callable``. A custom method.
            kwargs: See the parent arguments
        """
        self.default_subcommand = default_subcommand
        self.default_arguments = default_arguments or {}
        self.finetune = finetune

        model_class = make_args_optional(model_class, {"num_classes"})
        self.local_datamodule_class = datamodule_class
        super().__init__(model_class, datamodule_class=None, trainer_class=trainer_class, **kwargs)

    @contextlib.contextmanager
    def patch_default_subcommand(self):
        parse_common = self.parser._parse_common

        @functools.wraps(parse_common)
        def wrapper(cfg, *args, **kwargs):
            if not hasattr(cfg, "subcommand") or cfg['subcommand'] is None:
                cfg['subcommand'] = self.default_subcommand
            return parse_common(cfg, *args, **kwargs)

        self.parser._parse_common = wrapper
        yield
        self.parser._parse_common = parse_common

    def parse_arguments(self) -> None:
        with self.patch_default_subcommand():
            super().parse_arguments()

    def add_arguments_to_parser(self, parser) -> None:
        subcommands = parser.add_subcommands()
        self.add_from_method("from_folders", subcommands)
        self.add_from_method("from_csv", subcommands)

        parser.set_defaults(self.default_arguments)

    def add_from_method(self, method_name, subcommands):
        subcommand = ArgumentParser()
        subcommand.add_class_arguments(
            class_from_function(drop_kwargs(getattr(self.local_datamodule_class, method_name))), fail_untyped=False
        )
        subcommand.add_class_arguments(
            class_from_function(drop_kwargs(self.local_datamodule_class.preprocess_cls)),
            fail_untyped=False,
            skip={"train_transform", "val_transform", "test_transform", "predict_transform"}
        )
        subcommands.add_subcommand(method_name, subcommand)

    def instantiate_classes(self) -> None:
        """Instantiates the classes using settings from self.config"""
        sub_config = self.config.get("subcommand")
        self.datamodule = getattr(self.local_datamodule_class, sub_config)(**self.config.get(sub_config))
        self.config['model']['num_classes'] = self.datamodule.num_classes

        self.config_init = self.parser.instantiate_classes(self.config)
        self.model = self.config_init['model']
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
