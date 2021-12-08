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
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import pandas as pd
import torch
from pandas.core.frame import DataFrame

from flash.core.data.data_module import DataModule
from flash.core.data.data_pipeline import DataPipelineState
from flash.core.data.io.classification_input import ClassificationInput, ClassificationState
from flash.core.data.io.input import DataKeys, InputFormat
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.utilities.classification import TargetMode
from flash.core.data.utilities.paths import PATH_TYPE
from flash.core.integrations.labelstudio.input import _parse_labelstudio_arguments, LabelStudioTextClassificationInput
from flash.core.integrations.transformers.states import TransformersBackboneState
from flash.core.utilities.imports import _TEXT_AVAILABLE, requires
from flash.core.utilities.stages import RunningStage
from flash.text.input import TextDeserializer

if _TEXT_AVAILABLE:
    from datasets import Dataset, load_dataset
else:
    Dataset = object


class TextClassificationInput(ClassificationInput):
    @staticmethod
    def _resolve_target(target_keys: Union[str, List[str]], element: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(target_keys, List):
            element[DataKeys.TARGET] = element.pop(target_keys)
        else:
            element[DataKeys.TARGET] = [element[target_key] for target_key in target_keys]
        return element

    @requires("text")
    def load_data(
        self,
        hf_dataset: Dataset,
        input_key: str,
        target_keys: Optional[Union[str, List[str]]] = None,
        max_length: int = 128,
    ) -> Dataset:
        """Loads data into HuggingFace datasets.Dataset."""
        self.max_length = max_length

        if not self.predicting:
            hf_dataset = hf_dataset.map(partial(self._resolve_target, target_keys))
            targets = hf_dataset.to_dict()[DataKeys.TARGET]
            self.load_target_metadata(targets)

            # If we had binary multi-class targets then we also know the labels (column names)
            if self.target_mode is TargetMode.MULTI_BINARY and isinstance(target_keys, List):
                classification_state = self.get_state(ClassificationState)
                self.set_state(ClassificationState(target_keys, classification_state.num_classes))

        # remove extra columns
        extra_columns = set(hf_dataset.column_names) - {input_key, DataKeys.TARGET}
        hf_dataset = hf_dataset.remove_columns(extra_columns)

        if input_key != DataKeys.INPUT:
            hf_dataset = hf_dataset.rename_column(input_key, DataKeys.INPUT)

        return hf_dataset

    def load_sample(self, sample: Dict[str, Any]) -> Any:
        tokenized_sample = self.get_state(TransformersBackboneState).tokenizer(
            sample[DataKeys.INPUT], max_length=self.max_length, truncation=True, padding="max_length"
        )
        tokenized_sample = tokenized_sample.data
        if DataKeys.TARGET in sample:
            tokenized_sample[DataKeys.TARGET] = self.format_target(sample[DataKeys.TARGET])
        return tokenized_sample


class TextClassificationCSVInput(TextClassificationInput):
    @requires("text")
    def load_data(
        self,
        csv_file: PATH_TYPE,
        input_key: str,
        target_keys: Optional[Union[str, List[str]]] = None,
        max_length: int = 128,
    ) -> Dataset:
        dataset_dict = load_dataset("csv", data_files={"data": str(csv_file)})
        return super().load_data(dataset_dict["data"], input_key, target_keys, max_length)


class TextClassificationJSONInput(TextClassificationInput):
    @requires("text")
    def load_data(
        self,
        json_file: PATH_TYPE,
        field: str,
        input_key: str,
        target_keys: Optional[Union[str, List[str]]] = None,
        max_length: int = 128,
    ) -> Dataset:
        dataset_dict = load_dataset("json", data_files={"data": str(json_file)}, field=field)
        return super().load_data(dataset_dict["data"], input_key, target_keys, max_length)


class TextClassificationDataFrameInput(TextClassificationInput):
    @requires("text")
    def load_data(
        self,
        data_frame: pd.DataFrame,
        input_key: str,
        target_keys: Optional[Union[str, List[str]]] = None,
        max_length: int = 128,
    ) -> Dataset:
        return super().load_data(Dataset.from_pandas(data_frame), input_key, target_keys, max_length)


class TextClassificationParquetInput(TextClassificationInput):
    @requires("text")
    def load_data(
        self,
        parquet_file: PATH_TYPE,
        input_key: str,
        target_keys: Optional[Union[str, List[str]]] = None,
        max_length: int = 128,
    ) -> Dataset:
        return super().load_data(Dataset.from_parquet(str(parquet_file)), input_key, target_keys, max_length)


class TextClassificationListInput(TextClassificationInput):
    @requires("text")
    def load_data(
        self,
        inputs: List[str],
        targets: Optional[List[Any]] = None,
        max_length: int = 128,
    ) -> Dataset:
        if targets is not None:
            hf_dataset = Dataset.from_dict({DataKeys.INPUT: inputs, DataKeys.TARGET: targets})
        else:
            hf_dataset = Dataset.from_dict({DataKeys.INPUT: inputs})
        return super().load_data(hf_dataset, DataKeys.INPUT, DataKeys.TARGET, max_length)


class TextClassificationInputTransform(InputTransform):
    @requires("text")
    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        max_length: int = 128,
    ):
        self.max_length = max_length

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            inputs={
                InputFormat.CSV: TextClassificationCSVInput,
                InputFormat.JSON: TextClassificationJSONInput,
                InputFormat.PARQUET: TextClassificationParquetInput,
                InputFormat.HUGGINGFACE_DATASET: TextClassificationInput,
                InputFormat.DATAFRAME: TextClassificationDataFrameInput,
                InputFormat.LISTS: TextClassificationListInput,
            },
            default_input=InputFormat.LISTS,
            deserializer=TextDeserializer(max_length),
        )

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            **self.transforms,
            "max_length": self.max_length,
        }

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool):
        return cls(**state_dict)

    def per_sample_transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for key in sample:
            sample[key] = torch.as_tensor(sample[key])
        return sample


class TextClassificationData(DataModule):
    """Data Module for text classification tasks."""

    input_transform_cls = TextClassificationInputTransform

    @classmethod
    def from_csv(
        cls,
        input_field: str,
        target_fields: Optional[Union[str, Sequence[str]]] = None,
        train_file: Optional[PATH_TYPE] = None,
        val_file: Optional[PATH_TYPE] = None,
        test_file: Optional[PATH_TYPE] = None,
        predict_file: Optional[PATH_TYPE] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        max_length: int = 128,
        **data_module_kwargs: Any,
    ) -> "TextClassificationData":
        """Creates a :class:`~flash.text.classification.data.TextClassificationData` object from the given CSV
        files.

        Args:
            input_field: The field (column) in the pandas ``Dataset`` to use for the input.
            target_fields: The field or fields (columns) in the pandas ``Dataset`` to use for the target.
            train_file: The CSV file containing the training data.
            val_file: The CSV file containing the validation data.
            test_file: The CSV file containing the testing data.
            predict_file: The CSV file containing the data to use when predicting.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            max_length: The maximum sequence length.

        Returns:
            The constructed data module.
        """

        dataset_kwargs = dict(
            input_key=input_field,
            target_keys=target_fields,
            max_length=max_length,
            data_pipeline_state=DataPipelineState(),
        )

        return cls(
            TextClassificationCSVInput(RunningStage.TRAINING, train_file, **dataset_kwargs),
            TextClassificationCSVInput(RunningStage.VALIDATING, val_file, **dataset_kwargs),
            TextClassificationCSVInput(RunningStage.TESTING, test_file, **dataset_kwargs),
            TextClassificationCSVInput(RunningStage.PREDICTING, predict_file, **dataset_kwargs),
            input_transform=cls.input_transform_cls(
                train_transform,
                val_transform,
                test_transform,
                predict_transform,
                max_length=max_length,
            ),
            **data_module_kwargs,
        )

    @classmethod
    def from_json(
        cls,
        input_field: str,
        target_fields: Optional[Union[str, Sequence[str]]] = None,
        train_file: Optional[PATH_TYPE] = None,
        val_file: Optional[PATH_TYPE] = None,
        test_file: Optional[PATH_TYPE] = None,
        predict_file: Optional[PATH_TYPE] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        field: Optional[str] = None,
        max_length: int = 128,
        **data_module_kwargs: Any,
    ) -> "TextClassificationData":
        """Creates a :class:`~flash.text.classification.data.TextClassificationData` object from the given JSON
        files.

        Args:
            input_field: The field (column) in the pandas ``Dataset`` to use for the input.
            target_fields: The field or fields (columns) in the pandas ``Dataset`` to use for the target.
            train_file: The JSON file containing the training data.
            val_file: The JSON file containing the validation data.
            test_file: The JSON file containing the testing data.
            predict_file: The JSON file containing the data to use when predicting.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            field: To specify the field that holds the data in the JSON file.
            max_length: The maximum sequence length.

        Returns:
            The constructed data module.
        """

        dataset_kwargs = dict(
            input_key=input_field,
            target_keys=target_fields,
            field=field,
            max_length=max_length,
            data_pipeline_state=DataPipelineState(),
        )

        return cls(
            TextClassificationJSONInput(RunningStage.TRAINING, train_file, **dataset_kwargs),
            TextClassificationJSONInput(RunningStage.VALIDATING, val_file, **dataset_kwargs),
            TextClassificationJSONInput(RunningStage.TESTING, test_file, **dataset_kwargs),
            TextClassificationJSONInput(RunningStage.PREDICTING, predict_file, **dataset_kwargs),
            input_transform=cls.input_transform_cls(
                train_transform,
                val_transform,
                test_transform,
                predict_transform,
                max_length=max_length,
            ),
            **data_module_kwargs,
        )

    @classmethod
    def from_parquet(
        cls,
        input_field: str,
        target_fields: Optional[Union[str, Sequence[str]]] = None,
        train_file: Optional[PATH_TYPE] = None,
        val_file: Optional[PATH_TYPE] = None,
        test_file: Optional[PATH_TYPE] = None,
        predict_file: Optional[PATH_TYPE] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        max_length: int = 128,
        **data_module_kwargs: Any,
    ) -> "TextClassificationData":
        """Creates a :class:`~flash.text.classification.data.TextClassificationData` object from the given PARQUET
        files.

        Args:
            input_field: The field (column) in the pandas ``Dataset`` to use for the input.
            target_fields: The field or fields (columns) in the pandas ``Dataset`` to use for the target.
            train_file: The PARQUET file containing the training data.
            val_file: The PARQUET file containing the validation data.
            test_file: The PARQUET file containing the testing data.
            predict_file: The PARQUET file containing the data to use when predicting.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            max_length: The maximum sequence length.

        Returns:
            The constructed data module.
        """

        dataset_kwargs = dict(
            input_key=input_field,
            target_keys=target_fields,
            max_length=max_length,
            data_pipeline_state=DataPipelineState(),
        )

        return cls(
            TextClassificationParquetInput(RunningStage.TRAINING, train_file, **dataset_kwargs),
            TextClassificationParquetInput(RunningStage.VALIDATING, val_file, **dataset_kwargs),
            TextClassificationParquetInput(RunningStage.TESTING, test_file, **dataset_kwargs),
            TextClassificationParquetInput(RunningStage.PREDICTING, predict_file, **dataset_kwargs),
            input_transform=cls.input_transform_cls(
                train_transform,
                val_transform,
                test_transform,
                predict_transform,
                max_length=max_length,
            ),
            **data_module_kwargs,
        )

    @classmethod
    def from_hf_datasets(
        cls,
        input_field: str,
        target_fields: Optional[Union[str, Sequence[str]]] = None,
        train_hf_dataset: Optional[Dataset] = None,
        val_hf_dataset: Optional[Dataset] = None,
        test_hf_dataset: Optional[Dataset] = None,
        predict_hf_dataset: Optional[Dataset] = None,
        train_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        val_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        test_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        max_length: int = 128,
        **data_module_kwargs: Any,
    ) -> "TextClassificationData":
        """Creates a :class:`~flash.text.classification.data.TextClassificationData` object from the given Hugging
        Face datasets ``Dataset`` objects.

        Args:
            input_field: The field (column) in the pandas ``Dataset`` to use for the input.
            target_fields: The field or fields (columns) in the pandas ``Dataset`` to use for the target.
            train_hf_dataset: The pandas ``Dataset`` containing the training data.
            val_hf_dataset: The pandas ``Dataset`` containing the validation data.
            test_hf_dataset: The pandas ``Dataset`` containing the testing data.
            predict_hf_dataset: The pandas ``Dataset`` containing the data to use when predicting.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            max_length: The maximum sequence length.

        Returns:
            The constructed data module.
        """

        dataset_kwargs = dict(
            input_key=input_field,
            target_keys=target_fields,
            max_length=max_length,
            data_pipeline_state=DataPipelineState(),
        )

        return cls(
            TextClassificationInput(RunningStage.TRAINING, train_hf_dataset, **dataset_kwargs),
            TextClassificationInput(RunningStage.VALIDATING, val_hf_dataset, **dataset_kwargs),
            TextClassificationInput(RunningStage.TESTING, test_hf_dataset, **dataset_kwargs),
            TextClassificationInput(RunningStage.PREDICTING, predict_hf_dataset, **dataset_kwargs),
            input_transform=cls.input_transform_cls(
                train_transform,
                val_transform,
                test_transform,
                predict_transform,
                max_length=max_length,
            ),
            **data_module_kwargs,
        )

    @classmethod
    def from_data_frame(
        cls,
        input_field: str,
        target_fields: Optional[Union[str, Sequence[str]]] = None,
        train_data_frame: Optional[DataFrame] = None,
        val_data_frame: Optional[DataFrame] = None,
        test_data_frame: Optional[DataFrame] = None,
        predict_data_frame: Optional[DataFrame] = None,
        train_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        val_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        test_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        max_length: int = 128,
        **data_module_kwargs: Any,
    ) -> "TextClassificationData":
        """Creates a :class:`~flash.text.classification.data.TextClassificationData` object from the given pandas
        ``DataFrame`` objects.

        Args:
            input_field: The field (column) in the pandas ``DataFrame`` to use for the input.
            target_fields: The field or fields (columns) in the pandas ``DataFrame`` to use for the target.
            train_data_frame: The pandas ``DataFrame`` containing the training data.
            val_data_frame: The pandas ``DataFrame`` containing the validation data.
            test_data_frame: The pandas ``DataFrame`` containing the testing data.
            predict_data_frame: The pandas ``DataFrame`` containing the data to use when predicting.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            max_length: The maximum sequence length.

        Returns:
            The constructed data module.
        """

        dataset_kwargs = dict(
            input_key=input_field,
            target_keys=target_fields,
            max_length=max_length,
            data_pipeline_state=DataPipelineState(),
        )

        return cls(
            TextClassificationDataFrameInput(RunningStage.TRAINING, train_data_frame, **dataset_kwargs),
            TextClassificationDataFrameInput(RunningStage.VALIDATING, val_data_frame, **dataset_kwargs),
            TextClassificationDataFrameInput(RunningStage.TESTING, test_data_frame, **dataset_kwargs),
            TextClassificationDataFrameInput(RunningStage.PREDICTING, predict_data_frame, **dataset_kwargs),
            input_transform=cls.input_transform_cls(
                train_transform,
                val_transform,
                test_transform,
                predict_transform,
                max_length=max_length,
            ),
            **data_module_kwargs,
        )

    @classmethod
    def from_lists(
        cls,
        train_data: Optional[List[str]] = None,
        train_targets: Optional[Union[List[Any], List[List[Any]]]] = None,
        val_data: Optional[List[str]] = None,
        val_targets: Optional[Union[List[Any], List[List[Any]]]] = None,
        test_data: Optional[List[str]] = None,
        test_targets: Optional[Union[List[Any], List[List[Any]]]] = None,
        predict_data: Optional[List[str]] = None,
        train_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        val_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        test_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        max_length: int = 128,
        **data_module_kwargs: Any,
    ) -> "TextClassificationData":
        """Creates a :class:`~flash.text.classification.data.TextClassificationData` object from the given Python
        lists.

        Args:
            train_data: A list of sentences to use as the train inputs.
            train_targets: A list of targets to use as the train targets. For multi-label classification, the targets
                should be provided as a list of lists, where each inner list contains the targets for a sample.
            val_data: A list of sentences to use as the validation inputs.
            val_targets: A list of targets to use as the validation targets. For multi-label classification, the targets
                should be provided as a list of lists, where each inner list contains the targets for a sample.
            test_data: A list of sentences to use as the test inputs.
            test_targets: A list of targets to use as the test targets. For multi-label classification, the targets
                should be provided as a list of lists, where each inner list contains the targets for a sample.
            predict_data: A list of sentences to use when predicting.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            max_length: The maximum sequence length.

        Returns:
            The constructed data module.
        """

        dataset_kwargs = dict(max_length=max_length, data_pipeline_state=DataPipelineState())

        return cls(
            TextClassificationListInput(RunningStage.TRAINING, train_data, train_targets, **dataset_kwargs),
            TextClassificationListInput(RunningStage.VALIDATING, val_data, val_targets, **dataset_kwargs),
            TextClassificationListInput(RunningStage.TESTING, test_data, test_targets, **dataset_kwargs),
            TextClassificationListInput(RunningStage.PREDICTING, predict_data, **dataset_kwargs),
            input_transform=cls.input_transform_cls(
                train_transform,
                val_transform,
                test_transform,
                predict_transform,
                max_length=max_length,
            ),
            **data_module_kwargs,
        )

    @classmethod
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
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        val_split: Optional[float] = None,
        multi_label: Optional[bool] = False,
        max_length: int = 128,
        **data_module_kwargs: Any,
    ) -> "TextClassificationData":
        """Creates a :class:`~flash.core.data.data_module.DataModule` object
        from the given export file and data directory using the
        :class:`~flash.core.data.io.input.Input` of name
        :attr:`~flash.core.data.io.input.InputFormat.FOLDERS`
        from the passed or constructed :class:`~flash.core.data.io.input_transform.InputTransform`.

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
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            multi_label: Whether the labels are multi encoded.
            max_length: The maximum sequence length.
            data_module_kwargs: Additional keyword arguments to use when constructing the datamodule.

        Returns:
            The constructed data module.
        """

        train_data, val_data, test_data, predict_data = _parse_labelstudio_arguments(
            export_json=export_json,
            train_export_json=train_export_json,
            val_export_json=val_export_json,
            test_export_json=test_export_json,
            predict_export_json=predict_export_json,
            data_folder=data_folder,
            train_data_folder=train_data_folder,
            val_data_folder=val_data_folder,
            test_data_folder=test_data_folder,
            predict_data_folder=predict_data_folder,
            val_split=val_split,
            multi_label=multi_label,
        )

        dataset_kwargs = dict(data_pipeline_state=DataPipelineState(), max_length=max_length)

        return cls(
            LabelStudioTextClassificationInput(RunningStage.TRAINING, train_data, **dataset_kwargs),
            LabelStudioTextClassificationInput(RunningStage.VALIDATING, val_data, **dataset_kwargs),
            LabelStudioTextClassificationInput(RunningStage.TESTING, test_data, **dataset_kwargs),
            LabelStudioTextClassificationInput(RunningStage.PREDICTING, predict_data, **dataset_kwargs),
            input_transform=cls.input_transform_cls(
                train_transform,
                val_transform,
                test_transform,
                predict_transform,
                max_length=max_length,
            ),
            **data_module_kwargs,
        )
