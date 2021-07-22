from typing import Any, Type

import pytorch_lightning as pl
from jsonargparse import ArgumentParser

import flash
from flash.core.utilities.cli import class_from_function, LightningCLI


class FromFolders:

    def __init__(self, train_folder: str = "", val_folder: str = "", test_folder: str = ""):
        # self.train_folder = train_folder
        # self.val_folder = val_folder
        # self.test_folder = test_folder
        self.kwargs = {
            "train_folder": train_folder,
            "val_folder": val_folder,
            "test_folder": test_folder,
        }


class FlashCLI(LightningCLI):

    def __init__(
        self,
        model_class: Type[pl.LightningModule],
        datamodule_class: Type['flash.DataModule'],
        trainer_class: Type[pl.Trainer] = flash.Trainer,
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

        self.local_datamodule_class = datamodule_class
        # self.datamodule_class = self.get_datamodule_builder(datamodule_class)
        super().__init__(model_class, datamodule_class=None, trainer_class=trainer_class, **kwargs)
        # if isinstance(datasource, str):
        #     self.datasource = DefaultDataSources(datasource)
        # self.datasource_fn: Optional[Callable] = None

    def add_arguments_to_parser(self, parser) -> None:
        self.subcommand = ArgumentParser()
        self.subcommand.add_class_arguments(
            class_from_function(self.local_datamodule_class.from_folders), fail_untyped=False
        )

        subcommands = parser.add_subcommands()
        subcommands.add_subcommand('from_folders', self.subcommand)

    def instantiate_classes(self) -> None:
        """Instantiates the classes using settings from self.config"""
        self.config_init = self.parser.instantiate_classes(self.config)
        # print(self.config_init)
        sub_config = self.config_init.get("subcommand")
        # print(self.subcommand.instantiate_classes(self.config.get(sub_config)))
        self.datamodule = getattr(self.local_datamodule_class, sub_config)(**self.config_init.get(sub_config))
        self.model = self.config_init['model']
        self.instantiate_trainer()

    # def get_datamodule_builder(self, datamodule_class):
    #     def datamodule_builder(folders: FromFolders = lazy_instance(FromFolders)) -> flash.DataModule:
    #         # if datasource == DefaultDataSources.FOLDERS:
    #         #     datamodule_fn = datamodule_class.from_folders
    #         # elif datasource == DefaultDataSources.FILES:
    #         #     datamodule_fn = datamodule_class.from_files
    #         # elif datasource == DefaultDataSources.NUMPY:
    #         #     datamodule_fn = datamodule_class.from_numpy
    #         # elif datasource == DefaultDataSources.TENSORS:
    #         #     datamodule_fn = datamodule_class.from_tensor
    #         # elif datasource == DefaultDataSources.CSV:
    #         #     datamodule_fn = datamodule_class.from_csv
    #         # elif datasource == DefaultDataSources.JSON:
    #         #     datamodule_fn = datamodule_class.from_json
    #         # else:
    #         #     raise ValueError
    #         return datamodule_class.from_folders(**folders.kwargs)
    #     return datamodule_builder

    # def add_core_arguments_to_parser(self):
    #     """Adds arguments from the core classes to the parser"""
    #     self.parser.add_argument(
    #         'trainer_fn', type=FlashTrainerFn, default=self.trainer_fn, help='Trainer function to run'
    #     )
    #     self.parser.add_argument(
    #         '--seed_everything',
    #         type=Optional[int],
    #         default=self.seed_everything_default,
    #         help='Set to an int to run seed_everything with this value before classes instantiation',
    #     )
    #     self.parser.add_lightning_class_args(self.trainer_class, 'trainer')
    #     trainer_defaults = {'trainer.' + k: v for k, v in self.trainer_defaults.items() if k != 'callbacks'}
    #     self.parser.set_defaults(trainer_defaults)
    #     self.parser.add_lightning_class_args(self.model_class, 'model', subclass_mode=self.subclass_mode_model)

    # # Modification to work with `DataSource`s
    # if self.datasource is None:
    #     # Same as parent, use the `__init__`
    #     self.parser.add_lightning_class_args(self.datamodule_class, 'data', subclass_mode=self.subclass_mode_data)
    #     return
    #
    # if callable(self.datasource):
    #     # The user passed their own function, use it
    #     self.datasource_fn = self.datasource
    #
    # # Get the DataModule creation function with the data source provided
    # elif self.datasource == DefaultDataSources.FOLDERS:
    #     self.datasource_fn = self.datamodule_class.from_folders
    # elif self.datasource == DefaultDataSources.FILES:
    #     self.datasource_fn = self.datamodule_class.from_files
    # elif self.datasource == DefaultDataSources.NUMPY:
    #     self.datasource_fn = self.datamodule_class.from_numpy
    # elif self.datasource == DefaultDataSources.TENSORS:
    #     self.datasource_fn = self.datamodule_class.from_tensor
    # elif self.datasource == DefaultDataSources.CSV:
    #     self.datasource_fn = self.datamodule_class.from_csv
    # elif self.datasource == DefaultDataSources.JSON:
    #     self.datasource_fn = self.datamodule_class.from_json
    # else:
    #     raise ValueError
    #
    # self.parser.add_function_arguments(self.datasource_fn, nested_key='data')

    # def instantiate_datamodule(self) -> None:
    #     """Instantiates the datamodule using self.config_init['data'] if given"""
    #     if self.datasource_fn is not None:
    #         self.datamodule = self.datasource_fn(**self.config_init['data'])
    #     elif self.subclass_mode_data:
    #         self.datamodule = self.config_init['data']
    #     else:
    #         self.datamodule = self.datamodule_class(**self.config_init.get('data', {}))
