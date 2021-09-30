from typing import Any, Callable, Collection, Dict, Optional, Sequence, Type, TYPE_CHECKING, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from flash.core.data.auto_dataset import AutoDataset
from flash.core.data.data_source import DataSource, DefaultDataSources
from flash.core.data.process import Preprocess
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE, requires

if _FIFTYONE_AVAILABLE and TYPE_CHECKING:
    from fiftyone.core.collections import SampleCollection
else:
    SampleCollection = None


class AutoDatasetContainer:

    data_sources_registry: FlashRegistry

    def __init__(
        self,
        train_ds: Optional[AutoDataset] = None,
        val_ds: Optional[AutoDataset] = None,
        test_ds: Optional[AutoDataset] = None,
        predict_ds: Optional[AutoDataset] = None,
        data_source: Optional[DataSource] = None,
    ):
        self._train_ds = train_ds
        self._val_ds = val_ds
        self._test_ds = test_ds
        self._predict_ds = predict_ds
        self._data_source = data_source

    @classmethod
    def from_data_source(
        cls,
        enum,
        train_data,
        val_data,
        test_data,
        predict_data,
        **data_source_kwargs,
    ) -> "AutoDatasetContainer":
        data_source_cls: Type[DataSource] = cls.data_sources_registry.get(enum)
        data_source = data_source_cls(**data_source_kwargs)
        datasets = data_source.to_datasets(
            train_data,
            val_data,
            test_data,
            predict_data,
        )
        return cls(*datasets, data_source)

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[str] = None,
        val_folder: Optional[str] = None,
        test_folder: Optional[str] = None,
        predict_folder: Optional[str] = None,
        **data_source_kwargs,
    ) -> "AutoDatasetContainer":
        """Creates X.

        Args:
            train_folder: The folder containing the train data.
            val_folder: The folder containing the validation data.
            test_folder: The folder containing the test data.
            predict_folder: The folder containing the predict data.

        Returns:
            The constructed data module.
        """
        return cls.from_data_source(
            DefaultDataSources.FOLDERS,
            train_folder,
            val_folder,
            test_folder,
            predict_folder,
            **data_source_kwargs,
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
        **data_source_kwargs,
    ) -> "AutoDatasetContainer":
        """Creates a :class:`~flash.core.data.data_module.DataModule` object from the given sequences of files
        using the :class:`~flash.core.data.data_source.DataSource` of name
        :attr:`~flash.core.data.data_source.DefaultDataSources.FILES` from the passed or constructed
        :class:`~flash.core.data.process.Preprocess`.

        Args:
            train_files: A sequence of files to use as the train inputs.
            train_targets: A sequence of targets (one per train file) to use as the train targets.
            val_files: A sequence of files to use as the validation inputs.
            val_targets: A sequence of targets (one per validation file) to use as the validation targets.
            test_files: A sequence of files to use as the test inputs.
            test_targets: A sequence of targets (one per test file) to use as the test targets.
            predict_files: A sequence of files to use when predicting.

        Returns:
            The constructed data module.
        """
        return cls.from_data_source(
            DefaultDataSources.FILES,
            (train_files, train_targets),
            (val_files, val_targets),
            (test_files, test_targets),
            predict_files,
            **data_source_kwargs,
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
        **data_source_kwargs: Any,
    ) -> "AutoDatasetContainer":
        """Creates a X.

        Args:
            train_data: A tensor or collection of tensors to use as the train inputs.
            train_targets: A sequence of targets (one per train input) to use as the train targets.
            val_data: A tensor or collection of tensors to use as the validation inputs.
            val_targets: A sequence of targets (one per validation input) to use as the validation targets.
            test_data: A tensor or collection of tensors to use as the test inputs.
            test_targets: A sequence of targets (one per test input) to use as the test targets.
            predict_data: A tensor or collection of tensors to use when predicting.

        Returns:
            The constructed data module.

        Examples::

            data_module = AutoDatasetContainer.from_tensors(
                train_files=torch.rand(3, 128),
                train_targets=[1, 0, 1],
                train_transform={
                    "to_tensor_transform": torch.as_tensor,
                },
            )
        """
        return cls.from_data_source(
            DefaultDataSources.TENSORS,
            (train_data, train_targets),
            (val_data, val_targets),
            (test_data, test_targets),
            predict_data,
            **data_source_kwargs,
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
        **data_source_kwargs: Any,
    ) -> "AutoDatasetContainer":
        """Creates a X.

        Args:
            train_data: A numpy array to use as the train inputs.
            train_targets: A sequence of targets (one per train input) to use as the train targets.
            val_data: A numpy array to use as the validation inputs.
            val_targets: A sequence of targets (one per validation input) to use as the validation targets.
            test_data: A numpy array to use as the test inputs.
            test_targets: A sequence of targets (one per test input) to use as the test targets.
            predict_data: A numpy array to use when predicting.

        Returns:
            The constructed data module.

        Examples::

            data_module = DataModule.from_numpy(
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
            **data_source_kwargs,
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
        **data_source_kwargs: Any,
    ) -> "AutoDatasetContainer":
        """Creates a :class:`~flash.core.data.data_module.DataModule` object from the given JSON files using the
        :class:`~flash.core.data.data_source.DataSource`
        of name :attr:`~flash.core.data.data_source.DefaultDataSources.JSON`
        from the passed or constructed :class:`~flash.core.data.process.Preprocess`.

        Args:
            input_fields: The field or fields in the JSON objects to use for the input.
            target_fields: The field or fields in the JSON objects to use for the target.
            field: To specify the field that holds the data in the JSON file.
            train_file: The JSON file containing the training data.
            val_file: The JSON file containing the validation data.
            test_file: The JSON file containing the testing data.
            predict_file: The JSON file containing the data to use when predicting.

        Returns:
            The constructed data module.

        Examples::

            data_module = DataModule.from_json(
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

            data_module = DataModule.from_json(
                "input",
                "target",
                train_file="train_data.json",
                train_transform={
                    "to_tensor_transform": torch.as_tensor,
                },
                feild="data"
            )
        """
        return cls.from_data_source(
            DefaultDataSources.JSON,
            (train_file, input_fields, target_fields, field),
            (val_file, input_fields, target_fields, field),
            (test_file, input_fields, target_fields, field),
            (predict_file, input_fields, target_fields, field),
            **data_source_kwargs,
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
        **data_source_kwargs: Any,
    ) -> "AutoDatasetContainer":
        """Creates a :class:`~flash.core.data.data_module.DataModule` object from the given CSV files using the
        :class:`~flash.core.data.data_source.DataSource`
        of name :attr:`~flash.core.data.data_source.DefaultDataSources.CSV`
        from the passed or constructed :class:`~flash.core.data.process.Preprocess`.

        Args:
            input_fields: The field or fields (columns) in the CSV file to use for the input.
            target_fields: The field or fields (columns) in the CSV file to use for the target.
            train_file: The CSV file containing the training data.
            val_file: The CSV file containing the validation data.
            test_file: The CSV file containing the testing data.
            predict_file: The CSV file containing the data to use when predicting.

        Returns:
            The constructed data module.

        Examples::

            data_module = DataModule.from_csv(
                "input",
                "target",
                train_file="train_data.csv",
                train_transform={
                    "to_tensor_transform": torch.as_tensor,
                },
            )
        """
        return cls.from_data_source(
            DefaultDataSources.CSV,
            (train_file, input_fields, target_fields),
            (val_file, input_fields, target_fields),
            (test_file, input_fields, target_fields),
            (predict_file, input_fields, target_fields),
            **data_source_kwargs,
        )

    @classmethod
    def from_datasets(
        cls,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        predict_dataset: Optional[Dataset] = None,
        **data_source_kwargs: Any,
    ) -> "AutoDatasetContainer":
        """Creates a X.

        Args:
            train_dataset: Dataset used during training.
            val_dataset: Dataset used during validating.
            test_dataset: Dataset used during testing.
            predict_dataset: Dataset used during predicting.

        Returns:
            The constructed data module.

        Examples::

            data_module = DataModule.from_datasets(
                train_dataset=train_dataset,
                train_transform={
                    "to_tensor_transform": torch.as_tensor,
                },
            )
        """
        return cls.from_data_source(
            DefaultDataSources.DATASETS,
            train_dataset,
            val_dataset,
            test_dataset,
            predict_dataset,
            **data_source_kwargs,
        )

    @classmethod
    @requires("fiftyone")
    def from_fiftyone(
        cls,
        train_dataset: Optional[SampleCollection] = None,
        val_dataset: Optional[SampleCollection] = None,
        test_dataset: Optional[SampleCollection] = None,
        predict_dataset: Optional[SampleCollection] = None,
        **data_source_kwargs: Any,
    ) -> "AutoDatasetContainer":
        """Creates a X.

        Args:
            train_dataset: The ``fiftyone.core.collections.SampleCollection`` containing the train data.
            val_dataset: The ``fiftyone.core.collections.SampleCollection`` containing the validation data.
            test_dataset: The ``fiftyone.core.collections.SampleCollection`` containing the test data.
            predict_dataset: The ``fiftyone.core.collections.SampleCollection`` containing the predict data.

        Returns:
            The constructed data module.

        Examples::

            train_dataset = fo.Dataset.from_dir(
                "/path/to/dataset",
                dataset_type=fo.types.ImageClassificationDirectoryTree,
            )
            data_module = DataModule.from_fiftyone(
                train_data = train_dataset,
                train_transform={
                    "to_tensor_transform": torch.as_tensor,
                },
            )
        """
        return cls.from_data_source(
            DefaultDataSources.FIFTYONE,
            train_dataset,
            val_dataset,
            test_dataset,
            predict_dataset,
            **data_source_kwargs,
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
        **data_source_kwargs: Any,
    ) -> "AutoDatasetContainer":
        """Creates a :class:`~flash.core.data.data_module.DataModule` object
        from the given export file and data directory using the
        :class:`~flash.core.data.data_source.DataSource` of name
        :attr:`~flash.core.data.data_source.DefaultDataSources.FOLDERS`
        from the passed or constructed :class:`~flash.core.data.process.Preprocess`.

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

        Returns:
            The constructed data module.

        Examples::

            data_module = DataModule.from_labelstudio(
                export_json='project.json',
                data_folder='label-studio/media/upload',
                val_split=0.8,
            )
        """
        data = {
            "data_folder": data_folder,
            "export_json": export_json,
            "split": val_split,
            "multi_label": preprocess_kwargs.get("multi_label", False),
        }
        train_data = None
        val_data = None
        test_data = None
        predict_data = None
        if (train_data_folder or data_folder) and train_export_json:
            train_data = {
                "data_folder": train_data_folder or data_folder,
                "export_json": train_export_json,
                "multi_label": preprocess_kwargs.get("multi_label", False),
            }
        if (val_data_folder or data_folder) and val_export_json:
            val_data = {
                "data_folder": val_data_folder or data_folder,
                "export_json": val_export_json,
                "multi_label": preprocess_kwargs.get("multi_label", False),
            }
        if (test_data_folder or data_folder) and test_export_json:
            test_data = {
                "data_folder": test_data_folder or data_folder,
                "export_json": test_export_json,
                "multi_label": preprocess_kwargs.get("multi_label", False),
            }
        if (predict_data_folder or data_folder) and predict_export_json:
            predict_data = {
                "data_folder": predict_data_folder or data_folder,
                "export_json": predict_export_json,
                "multi_label": preprocess_kwargs.get("multi_label", False),
            }
        return cls.from_data_source(
            DefaultDataSources.LABELSTUDIO,
            train_data=train_data if train_data else data,
            val_data=val_data,
            test_data=test_data,
            predict_data=predict_data,
            **data_source_kwargs,
        )
