from typing import Any, Collection, List, Optional, Sequence, Type, TYPE_CHECKING, Union

import numpy as np
import torch
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.enums import LightningEnum
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import Dataset

from flash.core.data.data_source import DataSource, DefaultDataSources
from flash.core.data.flash_datasets import BaseDataset
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE, requires

if _FIFTYONE_AVAILABLE and TYPE_CHECKING:
    from fiftyone.core.collections import SampleCollection
else:
    SampleCollection = None


class FlashDatasetContainer:

    data_sources_registry: Optional[FlashRegistry] = None
    default_data_source: Optional[LightningEnum] = None

    def __init__(
        self,
        train_dataset: Optional[BaseDataset] = None,
        val_dataset: Optional[BaseDataset] = None,
        test_dataset: Optional[BaseDataset] = None,
        predict_dataset: Optional[BaseDataset] = None,
        **flash_dataset_kwargs: Any,
    ):
        """Container for AutoDataset.

        Args:
            data_source: Current data source used to create the datasets.
            train_dataset: The train dataset generated from the data_source.
            val_dataset: The val dataset generated from the data_source.
            test_dataset: The test dataset generated from the data_source.
            predict_dataset: The predict dataset generated from the data_source
            flash_dataset_kwargs: Kwargs used to generate the data source.
        """

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.predict_dataset = predict_dataset
        self._flash_dataset_kwargs = flash_dataset_kwargs

    @classmethod
    def available_constructors(cls) -> List[str]:
        return [name for name in dir(cls) if name.startswith("from_")]

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[str] = None,
        val_folder: Optional[str] = None,
        test_folder: Optional[str] = None,
        predict_folder: Optional[str] = None,
        **flash_data_kwargs,
    ) -> "FlashDatasetContainer":
        """Creates a :class:`~flash.core.data.loader.FlashDatasetContainer` object containing
        datasets generated from the given inputs to
        :meth:`~flash.core.data.flash_datasets.BaseDataset.load_data` (``train_data``, ``val_data``, ``test_data``,
        ``predict_data``).

        Args:
            train_folder: The folder containing the train data.
            val_folder: The folder containing the validation data.
            test_folder: The folder containing the test data.
            predict_folder: The folder containing the predict data.
            flash_data_kwargs: Kwargs used to generate the data source.

        Returns:
            The constructed auto dataset container.
        """
        return cls.from_data_source(
            DefaultDataSources.FOLDERS,
            train_folder,
            val_folder,
            test_folder,
            predict_folder,
            **flash_data_kwargs,
        )

    @classmethod
    def from_files(
        cls,
        train_files: Optional[Sequence[str]] = None,
        train_targets: Optional[Sequence[Any]] = None,
        val_files: Optional[Sequence[str]] = None,
        val_targets: Optional[Sequence[Any]] = None,
        test_files: Optional[Sequence[str]] = None,
        test_targets: Optional[Sequence[Any]] = None,
        predict_files: Optional[Sequence[str]] = None,
        **flash_data_kwargs,
    ) -> "FlashDatasetContainer":
        """Creates a :class:`~flash.core.data.loader.FlashDatasetContainer` object containing
        datasets generated from the given inputs to
        :meth:`~flash.core.data.flash_datasets.BaseDataset.load_data` (``train_data``, ``val_data``, ``test_data``,
        ``predict_data``).

        Args:
            train_files: A sequence of files to use as the train inputs.
            train_targets: A sequence of targets (one per train file) to use as the train targets.
            val_files: A sequence of files to use as the validation inputs.
            val_targets: A sequence of targets (one per validation file) to use as the validation targets.
            test_files: A sequence of files to use as the test inputs.
            test_targets: A sequence of targets (one per test file) to use as the test targets.
            predict_files: A sequence of files to use when predicting.
            flash_data_kwargs: Kwargs used to generate the data source.

        Returns:
            The constructed auto dataset container.
        """
        return cls.from_data_source(
            DefaultDataSources.FILES,
            (train_files, train_targets),
            (val_files, val_targets),
            (test_files, test_targets),
            predict_files,
            **flash_data_kwargs,
        )

    @classmethod
    def from_tensors(
        cls,
        train_data: Optional[Collection[torch.Tensor]] = None,
        train_targets: Optional[Collection[Any]] = None,
        val_data: Optional[Collection[torch.Tensor]] = None,
        val_targets: Optional[Sequence[Any]] = None,
        test_data: Optional[Collection[torch.Tensor]] = None,
        test_targets: Optional[Sequence[Any]] = None,
        predict_data: Optional[Collection[torch.Tensor]] = None,
        **flash_data_kwargs: Any,
    ) -> "FlashDatasetContainer":
        """Creates a :class:`~flash.core.data.loader.FlashDatasetContainer` object containing
        datasets generated from the given inputs to
        :meth:`~flash.core.data.flash_datasets.BaseDataset.load_data` (``train_data``, ``val_data``, ``test_data``,
        ``predict_data``).

        Args:
            train_data: A tensor or collection of tensors to use as the train inputs.
            train_targets: A sequence of targets (one per train input) to use as the train targets.
            val_data: A tensor or collection of tensors to use as the validation inputs.
            val_targets: A sequence of targets (one per validation input) to use as the validation targets.
            test_data: A tensor or collection of tensors to use as the test inputs.
            test_targets: A sequence of targets (one per test input) to use as the test targets.
            predict_data: A tensor or collection of tensors to use when predicting.
            flash_data_kwargs: Kwargs used to generate the data source.

        Returns:
            The constructed auto dataset container.
        """
        return cls.from_data_source(
            DefaultDataSources.TENSORS,
            (train_data, train_targets),
            (val_data, val_targets),
            (test_data, test_targets),
            predict_data,
            **flash_data_kwargs,
        )

    @classmethod
    def from_numpy(
        cls,
        train_data: Optional[Collection[np.ndarray]] = None,
        train_targets: Optional[Collection[Any]] = None,
        val_data: Optional[Collection[np.ndarray]] = None,
        val_targets: Optional[Sequence[Any]] = None,
        test_data: Optional[Collection[np.ndarray]] = None,
        test_targets: Optional[Sequence[Any]] = None,
        predict_data: Optional[Collection[np.ndarray]] = None,
        **flash_data_kwargs: Any,
    ) -> "FlashDatasetContainer":
        """Creates a :class:`~flash.core.data.loader.FlashDatasetContainer` object containing
        datasets generated from the given inputs to
        :meth:`~flash.core.data.flash_datasets.BaseDataset.load_data` (``train_data``, ``val_data``, ``test_data``,
        ``predict_data``).

        Args:
            train_data: A numpy array to use as the train inputs.
            train_targets: A sequence of targets (one per train input) to use as the train targets.
            val_data: A numpy array to use as the validation inputs.
            val_targets: A sequence of targets (one per validation input) to use as the validation targets.
            test_data: A numpy array to use as the test inputs.
            test_targets: A sequence of targets (one per test input) to use as the test targets.
            predict_data: A numpy array to use when predicting.
            flash_data_kwargs: Kwargs used to generate the data source.

        Returns:
            The constructed auto dataset container.

        Examples::

            container = AutoContainer.from_numpy(
                train_files=np.random.rand(3, 128),
                train_targets=[1, 0, 1],
                train_transform={
                    "to_tensor_transform": torch.as_tensor,
                },
            )
        """
        return cls.from_data_source(
            DefaultDataSources.NUMPY,
            (train_data, train_targets),
            (val_data, val_targets),
            (test_data, test_targets),
            predict_data,
            **flash_data_kwargs,
        )

    @classmethod
    def from_json(
        cls,
        input_fields: Union[str, Sequence[str]],
        target_fields: Optional[Union[str, Sequence[str]]] = None,
        field: Optional[str] = None,
        train_file: Optional[str] = None,
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        predict_file: Optional[str] = None,
        **flash_data_kwargs: Any,
    ) -> "FlashDatasetContainer":
        """Creates a :class:`~flash.core.data.loader.FlashDatasetContainer` object containing
        datasets generated from the given inputs to
        :meth:`~flash.core.data.flash_datasets.BaseDataset.load_data` (``train_data``, ``val_data``, ``test_data``,
        ``predict_data``).

        Args:
            input_fields: The field or fields in the JSON objects to use for the input.
            target_fields: The field or fields in the JSON objects to use for the target.
            field: To specify the field that holds the data in the JSON file.
            train_file: The JSON file containing the training data.
            val_file: The JSON file containing the validation data.
            test_file: The JSON file containing the testing data.
            predict_file: The JSON file containing the data to use when predicting.
            flash_data_kwargs: Kwargs used to generate the data source.

        Returns:
            The constructed auto dataset container.

        Examples::

            container = AutoContainer.from_json(
                "input",
                "target",
                train_file="train_data.json",
                train_transform={
                    "to_tensor_transform": torch.as_tensor,
                },
            )

            # In the case where the data is of the form:
            # {
            #     "version": 0.0.x,
            #     "data": [
            #         {
            #             "input_field" : "input_data",
            #             "target_field" : "target_output"
            #         },
            #         ...
            #     ]
            # }

            container = AutoContainer.from_json(
                "input",
                "target",
                train_file="train_data.json",
                field="data"
            )
        """
        return cls.from_data_source(
            DefaultDataSources.JSON,
            (train_file, input_fields, target_fields, field),
            (val_file, input_fields, target_fields, field),
            (test_file, input_fields, target_fields, field),
            (predict_file, input_fields, target_fields, field),
            **flash_data_kwargs,
        )

    @classmethod
    def from_csv(
        cls,
        input_fields: Union[str, Sequence[str]],
        target_fields: Optional[Union[str, Sequence[str]]] = None,
        train_file: Optional[str] = None,
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        predict_file: Optional[str] = None,
        **flash_data_kwargs: Any,
    ) -> "FlashDatasetContainer":
        """Creates a :class:`~flash.core.data.loader.FlashDatasetContainer` object containing
        datasets generated from the given inputs to
        :meth:`~flash.core.data.flash_datasets.BaseDataset.load_data` (``train_data``, ``val_data``, ``test_data``,
        ``predict_data``).

        Args:
            input_fields: The field or fields (columns) in the CSV file to use for the input.
            target_fields: The field or fields (columns) in the CSV file to use for the target.
            train_file: The CSV file containing the training data.
            val_file: The CSV file containing the validation data.
            test_file: The CSV file containing the testing data.
            predict_file: The CSV file containing the data to use when predicting.
            flash_data_kwargs: Kwargs used to generate the data source.

        Returns:
            The constructed auto dataset container.

        Examples::

            container = AutoContainer.from_csv(
                "input",
                "target",
                train_file="train_data.csv",
            )
        """
        return cls.from_data_source(
            DefaultDataSources.CSV,
            (train_file, input_fields, target_fields),
            (val_file, input_fields, target_fields),
            (test_file, input_fields, target_fields),
            (predict_file, input_fields, target_fields),
            **flash_data_kwargs,
        )

    @classmethod
    def from_datasets(
        cls,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        predict_dataset: Optional[Dataset] = None,
        **flash_data_kwargs: Any,
    ) -> "FlashDatasetContainer":
        """Creates a :class:`~flash.core.data.loader.FlashDatasetContainer` object containing
        datasets generated from the given inputs to
        :meth:`~flash.core.data.flash_datasets.BaseDataset.load_data` (``train_data``, ``val_data``, ``test_data``,
        ``predict_data``).

        Args:
            train_dataset: Dataset used during training.
            val_dataset: Dataset used during validating.
            test_dataset: Dataset used during testing.
            predict_dataset: Dataset used during predicting.
            flash_data_kwargs: Kwargs used to generate the data source.

        Returns:
            The constructed auto dataset container.

        Examples::

            container = AutoContainer.from_csv(
                train_dataset=train_dataset,
            )
        """
        return cls.from_data_source(
            DefaultDataSources.DATASETS,
            train_dataset,
            val_dataset,
            test_dataset,
            predict_dataset,
            **flash_data_kwargs,
        )

    @classmethod
    @requires("fiftyone")
    def from_fiftyone(
        cls,
        train_dataset: Optional[SampleCollection] = None,
        val_dataset: Optional[SampleCollection] = None,
        test_dataset: Optional[SampleCollection] = None,
        predict_dataset: Optional[SampleCollection] = None,
        **flash_data_kwargs: Any,
    ) -> "FlashDatasetContainer":
        """Creates a :class:`~flash.core.data.loader.FlashDatasetContainer` object containing
        datasets generated from the given inputs to
        :meth:`~flash.core.data.flash_datasets.BaseDataset.load_data` (``train_data``, ``val_data``, ``test_data``,
        ``predict_data``).

        Args:
            train_dataset: The ``fiftyone.core.collections.SampleCollection`` containing the train data.
            val_dataset: The ``fiftyone.core.collections.SampleCollection`` containing the validation data.
            test_dataset: The ``fiftyone.core.collections.SampleCollection`` containing the test data.
            predict_dataset: The ``fiftyone.core.collections.SampleCollection`` containing the predict data.
            flash_data_kwargs: Kwargs used to generate the data source.

        Returns:
            The constructed auto dataset container.

        Examples::

            train_dataset = fo.Dataset.from_dir(
                "/path/to/dataset",
                dataset_type=fo.types.ImageClassificationDirectoryTree,
            )
            container = AutoContainer.from_fiftyone(
                train_data = train_dataset,
            )
        """
        return cls.from_data_source(
            DefaultDataSources.FIFTYONE,
            train_dataset,
            val_dataset,
            test_dataset,
            predict_dataset,
            **flash_data_kwargs,
        )

    @classmethod
    @requires("labelstudio")
    def from_labelstudio(
        cls,
        export_json: str = None,
        train_export_json: str = None,
        val_export_json: str = None,
        test_export_json: str = None,
        predict_export_json: str = None,
        data_folder: str = None,
        train_data_folder: str = None,
        val_data_folder: str = None,
        test_data_folder: str = None,
        predict_data_folder: str = None,
        **flash_data_kwargs: Any,
    ) -> "FlashDatasetContainer":
        """Creates a :class:`~flash.core.data.loader.FlashDatasetContainer` object containing
        datasets generated from the given inputs to
        :meth:`~flash.core.data.flash_datasets.BaseDataset.load_data` (``train_data``, ``val_data``, ``test_data``,
        ``predict_data``).

        Args:
            export_json: path to label studio export file
            train_export_json: path to label studio export file for train set,
            overrides export_json if specified
            val_export_json: path to label studio export file for validation
            test_export_json: path to label studio export file for test
            predict_export_json: path to label studio export file for predict
            data_folder: path to label studio data folder
            train_data_folder: path to label studio data folder for train data set,
            overrides data_folder if specified
            val_data_folder: path to label studio data folder for validation data
            test_data_folder: path to label studio data folder for test data
            predict_data_folder: path to label studio data folder for predict data
            flash_data_kwargs: Kwargs used to generate the data source.

        Returns:
            The constructed auto dataset container.

        Examples::

            container = AutoContainer.from_labelstudio(
                export_json='project.json',
                data_folder='label-studio/media/upload',
            )
        """
        data = {
            "data_folder": data_folder,
            "export_json": export_json,
            "split": flash_data_kwargs.get("val_split", 0.0),
            "multi_label": flash_data_kwargs.get("multi_label", False),
        }
        train_data = None
        val_data = None
        test_data = None
        predict_data = None
        if (train_data_folder or data_folder) and train_export_json:
            train_data = {
                "data_folder": train_data_folder or data_folder,
                "export_json": train_export_json,
                "multi_label": flash_data_kwargs.get("multi_label", False),
            }
        if (val_data_folder or data_folder) and val_export_json:
            val_data = {
                "data_folder": val_data_folder or data_folder,
                "export_json": val_export_json,
                "multi_label": flash_data_kwargs.get("multi_label", False),
            }
        if (test_data_folder or data_folder) and test_export_json:
            test_data = {
                "data_folder": test_data_folder or data_folder,
                "export_json": test_export_json,
                "multi_label": flash_data_kwargs.get("multi_label", False),
            }
        if (predict_data_folder or data_folder) and predict_export_json:
            predict_data = {
                "data_folder": predict_data_folder or data_folder,
                "export_json": predict_export_json,
                "multi_label": flash_data_kwargs.get("multi_label", False),
            }
        return cls.from_data_source(
            DefaultDataSources.LABELSTUDIO,
            train_data=train_data if train_data else data,
            val_data=val_data,
            test_data=test_data,
            predict_data=predict_data,
            **flash_data_kwargs,
        )

    @classmethod
    def from_data_source(
        cls,
        enum: LightningEnum,
        train_data: Optional[Any] = None,
        val_data: Optional[Any] = None,
        test_data: Optional[Any] = None,
        predict_data: Optional[Any] = None,
        **flash_dataset_kwargs,
    ) -> "FlashDatasetContainer":
        cls._verify_container(enum)
        flash_dataset_cls: BaseDataset = cls.data_sources_registry.get(enum)
        return cls(
            cls._flash_dataset_creation(flash_dataset_cls, train_data, RunningStage.TRAINING, **flash_dataset_kwargs),
            cls._flash_dataset_creation(flash_dataset_cls, val_data, RunningStage.VALIDATING, **flash_dataset_kwargs),
            cls._flash_dataset_creation(flash_dataset_cls, test_data, RunningStage.TESTING, **flash_dataset_kwargs),
            cls._flash_dataset_creation(
                flash_dataset_cls, predict_data, RunningStage.PREDICTING, **flash_dataset_kwargs
            ),
        )

    @staticmethod
    def _flash_dataset_creation(flash_dataset_cls, data, running_state, **kwargs) -> Optional[BaseDataset]:
        if data is not None:
            return flash_dataset_cls.from_data(data, running_state, **kwargs)

    @classmethod
    def _verify_container(cls, enum: LightningEnum) -> None:
        if (
            not cls.data_sources_registry
            or not cls.default_data_source
            or not isinstance(cls.data_sources_registry, FlashRegistry)
            or not isinstance(cls.default_data_source, LightningEnum)
        ):
            raise MisconfigurationException(
                "The ``AutoContainer`` should have ``data_sources_registry`` (FlashRegistry) populated "
                "with datasource class and ``default_data_source`` (LightningEnum) class attributes. "
            )

        if enum not in cls.data_sources_registry.available_keys():
            available_constructors = [f"from_{key.name.lower()}" for key in cls.data_sources_registry.available_keys()]
            raise MisconfigurationException(
                f"The ``AutoContainer`` ``data_sources_registry`` doesn't contain the associated {enum} "
                f"HINT: Here are the available constructors {available_constructors}"
            )

    @classmethod
    def register_data_source(cls, data_source_cls: Type[DataSource], enum: LightningEnum) -> None:
        if cls.data_sources_registry is None:
            raise MisconfigurationException("The class attribute `data_sources_registry` should be set. ")
        cls.data_sources_registry(fn=data_source_cls, name=enum)
