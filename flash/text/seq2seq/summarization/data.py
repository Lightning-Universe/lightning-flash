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
from typing import Any, Dict, List, Optional, Type, Union

from flash.core.data.data_module import DataModule
from flash.core.data.data_pipeline import DataPipelineState
from flash.core.data.io.input import Input
from flash.core.data.utilities.paths import PATH_TYPE
from flash.core.integrations.transformers.input_transform import TransformersInputTransform
from flash.core.utilities.imports import _TEXT_AVAILABLE
from flash.core.utilities.stages import RunningStage
from flash.core.utilities.types import INPUT_TRANSFORM_TYPE
from flash.text.seq2seq.core.input import Seq2SeqCSVInput, Seq2SeqInputBase, Seq2SeqJSONInput, Seq2SeqListInput
from flash.text.seq2seq.core.output_transform import Seq2SeqOutputTransform

if _TEXT_AVAILABLE:
    from datasets import Dataset
else:
    Dataset = object


class SummarizationData(DataModule):
    """DataModule for Summarization tasks."""

    input_transform_cls = TransformersInputTransform
    output_transform_cls = Seq2SeqOutputTransform

    @classmethod
    def from_csv(
        cls,
        input_field: str,
        target_field: Optional[str] = None,
        train_file: Optional[PATH_TYPE] = None,
        val_file: Optional[PATH_TYPE] = None,
        test_file: Optional[PATH_TYPE] = None,
        predict_file: Optional[PATH_TYPE] = None,
        train_transform: INPUT_TRANSFORM_TYPE = TransformersInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = TransformersInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = TransformersInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = TransformersInputTransform,
        input_cls: Type[Input] = Seq2SeqCSVInput,
        transform_kwargs: Optional[Dict] = None,
        max_source_length: int = 128,
        max_target_length: int = 128,
        padding: Union[str, bool] = "max_length",
        **data_module_kwargs: Any,
    ) -> "SummarizationData":
        """Load the :class:`~flash.text.seq2seq.summarization.data.SummarizationData` from CSV files containing
        input text snippets and their corresponding target text snippets.

        Input text snippets will be extracted from the ``input_field`` column in the CSV files.
        Target text snippets will be extracted from the ``target_field`` column in the CSV files.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            input_field: The field (column name) in the CSV files containing the input text snippets.
            target_field: The field (column name) in the CSV files containing the target text snippets.
            train_file: The CSV file to use when training.
            val_file: The CSV file to use when validating.
            test_file: The CSV file to use when testing.
            predict_file: The CSV file to use when predicting.
            train_transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use when training.
            val_transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use when validating.
            test_transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use when testing.
            predict_transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use when
                predicting.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            max_source_length: The maximum length to pad / truncate input sequences to.
            max_target_length: The maximum length to pad / truncate target sequences to.
            padding: The type of padding to apply. One of: "longest" or ``True``, "max_length", "do_not_pad" or
                ``False``.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.text.classification.data.TextClassificationData`.

        Examples
        ________

        .. testsetup::

            >>> import os
            >>> from pandas import DataFrame
            >>> DataFrame.from_dict({
            ...     "texts": ["A long paragraph", "A news article"],
            ...     "summaries": ["A short paragraph", "A news headline"],
            ... }).to_csv("train_data.csv", index=False)
            >>> DataFrame.from_dict({
            ...     "texts": ["A movie review", "A book chapter"],
            ... }).to_csv("predict_data.csv", index=False)

        The file ``train_data.csv`` contains the following:

        .. code-block::

            texts,summaries
            A long paragraph,A short paragraph
            A news article,A news headline

        The file ``predict_data.csv`` contains the following:

        .. code-block::

            texts
            A movie review
            A book chapter

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.text import SummarizationTask, SummarizationData
            >>> datamodule = SummarizationData.from_csv(
            ...     "texts",
            ...     "summaries",
            ...     train_file="train_data.csv",
            ...     predict_file="predict_data.csv",
            ...     batch_size=2,
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Downloading...
            >>> model = SummarizationTask()
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> os.remove("train_data.csv")
            >>> os.remove("predict_data.csv")
        """

        ds_kw = dict(
            input_key=input_field,
            target_key=target_field,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_file, transform=train_transform, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_file, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_file, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_file, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )

    @classmethod
    def from_json(
        cls,
        input_field: str,
        target_field: Optional[str] = None,
        train_file: Optional[PATH_TYPE] = None,
        val_file: Optional[PATH_TYPE] = None,
        test_file: Optional[PATH_TYPE] = None,
        predict_file: Optional[PATH_TYPE] = None,
        train_transform: INPUT_TRANSFORM_TYPE = TransformersInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = TransformersInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = TransformersInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = TransformersInputTransform,
        input_cls: Type[Input] = Seq2SeqJSONInput,
        transform_kwargs: Optional[Dict] = None,
        field: Optional[str] = None,
        max_source_length: int = 128,
        max_target_length: int = 128,
        padding: Union[str, bool] = "max_length",
        **data_module_kwargs: Any,
    ) -> "SummarizationData":
        """Creates a :class:`~flash.text.seq2seq.core.data.Seq2SeqData` object from the given JSON files.

        Args:
            input_field: The field (column) in the JSON file to use for the input.
            target_field: The field (column) in the JSON file to use for the target.
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
            max_source_length: The maximum source sequence length.
            max_target_length: The maximum target sequence length.
            padding: The padding mode to use.

        Returns:
            The constructed data module.
        """

        ds_kw = dict(
            input_key=input_field,
            target_key=target_field,
            field=field,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_file, transform=train_transform, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_file, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_file, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_file, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )

    @classmethod
    def from_hf_datasets(
        cls,
        input_field: str,
        target_field: Optional[str] = None,
        train_hf_dataset: Optional[Dataset] = None,
        val_hf_dataset: Optional[Dataset] = None,
        test_hf_dataset: Optional[Dataset] = None,
        predict_hf_dataset: Optional[Dataset] = None,
        train_transform: INPUT_TRANSFORM_TYPE = TransformersInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = TransformersInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = TransformersInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = TransformersInputTransform,
        input_cls: Type[Input] = Seq2SeqInputBase,
        transform_kwargs: Optional[Dict] = None,
        max_source_length: int = 128,
        max_target_length: int = 128,
        padding: Union[str, bool] = "max_length",
        **data_module_kwargs: Any,
    ) -> "SummarizationData":
        """Creates a :class:`~flash.text.seq2seq.core.data.Seq2SeqData` object from the given Hugging Face datasets
        ``Dataset`` objects.

        Args:
            input_field: The field (column) in the ``Dataset`` to use for the input.
            target_field: The field (column) in the ``Dataset`` to use for the target.
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
            max_source_length: The maximum source sequence length.
            max_target_length: The maximum target sequence length.
            padding: The padding mode to use.

        Returns:
            The constructed data module.
        """

        ds_kw = dict(
            input_key=input_field,
            target_key=target_field,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_hf_dataset, transform=train_transform, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_hf_dataset, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_hf_dataset, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_hf_dataset, transform=predict_transform, **ds_kw),
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
        train_transform: INPUT_TRANSFORM_TYPE = TransformersInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = TransformersInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = TransformersInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = TransformersInputTransform,
        input_cls: Type[Input] = Seq2SeqListInput,
        transform_kwargs: Optional[Dict] = None,
        max_source_length: int = 128,
        max_target_length: int = 128,
        padding: Union[str, bool] = "max_length",
        **data_module_kwargs: Any,
    ) -> "SummarizationData":
        """Creates a :class:`~flash.text.seq2seq.core.data.Seq2SeqData` object from the given Python lists.

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
            max_source_length: The maximum source sequence length.
            max_target_length: The maximum target sequence length.
            padding: The padding mode to use.

        Returns:
            The constructed data module.
        """

        ds_kw = dict(
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_data, train_targets, transform=train_transform, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_data, val_targets, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_data, test_targets, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_data, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )
