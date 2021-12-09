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
from typing import Any, Dict, Optional, Type, Union

from flash.core.data.data_pipeline import DataPipelineState
from flash.core.data.io.input import Input
from flash.core.data.new_data_module import DataModule
from flash.core.data.utilities.paths import PATH_TYPE
from flash.core.utilities.stages import RunningStage
from flash.core.utilities.types import INPUT_TRANSFORM_TYPE
from flash.text.question_answering.input import (
    QuestionAnsweringCSVInput,
    QuestionAnsweringDictionaryInput,
    QuestionAnsweringJSONInput,
    QuestionAnsweringSQuADInput,
)
from flash.text.question_answering.input_transform import QuestionAnsweringInputTransform
from flash.text.question_answering.output_transform import QuestionAnsweringOutputTransform


class QuestionAnsweringData(DataModule):
    """Data module for QuestionAnswering task."""

    input_transform_cls = QuestionAnsweringInputTransform
    output_transform_cls = QuestionAnsweringOutputTransform

    @classmethod
    def from_csv(
        cls,
        train_file: Optional[PATH_TYPE] = None,
        val_file: Optional[PATH_TYPE] = None,
        test_file: Optional[PATH_TYPE] = None,
        predict_file: Optional[PATH_TYPE] = None,
        train_transform: INPUT_TRANSFORM_TYPE = QuestionAnsweringInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = QuestionAnsweringInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = QuestionAnsweringInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = QuestionAnsweringInputTransform,
        input_cls: Type[Input] = QuestionAnsweringCSVInput,
        transform_kwargs: Optional[Dict] = None,
        max_source_length: int = 384,
        max_target_length: int = 30,
        padding: Union[str, bool] = "max_length",
        question_column_name: str = "question",
        context_column_name: str = "context",
        answer_column_name: str = "answer",
        doc_stride: int = 128,
        **data_module_kwargs: Any,
    ) -> "QuestionAnsweringData":
        """Creates a :class:`~flash.text.question_answering.data.QuestionAnsweringData` object from the given CSV
        files.

        Args:
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
            max_source_length: Max length of the sequence to be considered during tokenization.
            max_target_length: Max length of each answer to be produced.
            padding: Padding type during tokenization.
            question_column_name: The key in the JSON file to recognize the question field.
            context_column_name: The key in the JSON file to recognize the context field.
            answer_column_name: The key in the JSON file to recognize the answer field.
            doc_stride: The stride amount to be taken when splitting up a long document into chunks.

        Returns:
            The constructed data module.
        """

        ds_kw = dict(
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
            question_column_name=question_column_name,
            context_column_name=context_column_name,
            answer_column_name=answer_column_name,
            doc_stride=doc_stride,
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
        train_file: Optional[PATH_TYPE] = None,
        val_file: Optional[PATH_TYPE] = None,
        test_file: Optional[PATH_TYPE] = None,
        predict_file: Optional[PATH_TYPE] = None,
        train_transform: INPUT_TRANSFORM_TYPE = QuestionAnsweringInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = QuestionAnsweringInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = QuestionAnsweringInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = QuestionAnsweringInputTransform,
        input_cls: Type[Input] = QuestionAnsweringJSONInput,
        transform_kwargs: Optional[Dict] = None,
        field: Optional[str] = None,
        max_source_length: int = 384,
        max_target_length: int = 30,
        padding: Union[str, bool] = "max_length",
        question_column_name: str = "question",
        context_column_name: str = "context",
        answer_column_name: str = "answer",
        doc_stride: int = 128,
        **data_module_kwargs: Any,
    ) -> "QuestionAnsweringData":
        """Creates a :class:`~flash.text.question_answering.data.QuestionAnsweringData` object from the given JSON
        files.

        Args:
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
            field: The field that holds the data in the JSON file.
            max_source_length: Max length of the sequence to be considered during tokenization.
            max_target_length: Max length of each answer to be produced.
            padding: Padding type during tokenization.
            question_column_name: The key in the JSON file to recognize the question field.
            context_column_name: The key in the JSON file to recognize the context field.
            answer_column_name: The key in the JSON file to recognize the answer field.
            doc_stride: The stride amount to be taken when splitting up a long document into chunks.

        Returns:
            The constructed data module.
        """

        ds_kw = dict(
            field=field,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
            question_column_name=question_column_name,
            context_column_name=context_column_name,
            answer_column_name=answer_column_name,
            doc_stride=doc_stride,
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
    def from_squad_v2(
        cls,
        train_file: Optional[str] = None,
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        predict_file: Optional[str] = None,
        train_transform: INPUT_TRANSFORM_TYPE = QuestionAnsweringInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = QuestionAnsweringInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = QuestionAnsweringInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = QuestionAnsweringInputTransform,
        input_cls: Type[Input] = QuestionAnsweringSQuADInput,
        transform_kwargs: Optional[Dict] = None,
        max_source_length: int = 384,
        max_target_length: int = 30,
        padding: Union[str, bool] = "max_length",
        question_column_name: str = "question",
        context_column_name: str = "context",
        answer_column_name: str = "answer",
        doc_stride: int = 128,
        **data_module_kwargs: Any,
    ) -> "QuestionAnsweringData":
        """Creates a :class:`~flash.text.question_answering.data.QuestionAnsweringData` object from the given data
        JSON files in the SQuAD2.0 format.

        Args:
            train_file: The JSON file containing the training data.
            val_file: The JSON file containing the validation data.
            test_file: The JSON file containing the testing data.
            predict_file: The JSON file containing the predict data.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            max_source_length: Max length of the sequence to be considered during tokenization.
            max_target_length: Max length of each answer to be produced.
            padding: Padding type during tokenization.
            question_column_name: The key in the JSON file to recognize the question field.
            context_column_name: The key in the JSON file to recognize the context field.
            answer_column_name: The key in the JSON file to recognize the answer field.
            doc_stride: The stride amount to be taken when splitting up a long document into chunks.

        Returns:
            The constructed data module.
        """

        ds_kw = dict(
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
            question_column_name=question_column_name,
            context_column_name=context_column_name,
            answer_column_name=answer_column_name,
            doc_stride=doc_stride,
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
    def from_dicts(
        cls,
        train_data: Optional[Dict[str, Any]] = None,
        val_data: Optional[Dict[str, Any]] = None,
        test_data: Optional[Dict[str, Any]] = None,
        predict_data: Optional[Dict[str, Any]] = None,
        train_transform: INPUT_TRANSFORM_TYPE = QuestionAnsweringInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = QuestionAnsweringInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = QuestionAnsweringInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = QuestionAnsweringInputTransform,
        input_cls: Type[Input] = QuestionAnsweringDictionaryInput,
        transform_kwargs: Optional[Dict] = None,
        max_source_length: int = 384,
        max_target_length: int = 30,
        padding: Union[str, bool] = "max_length",
        question_column_name: str = "question",
        context_column_name: str = "context",
        answer_column_name: str = "answer",
        doc_stride: int = 128,
        **data_module_kwargs: Any,
    ) -> "QuestionAnsweringData":
        """Creates a :class:`~flash.text.question_answering.data.QuestionAnsweringData` object from the given data
        dictionaries.

        Args:
            train_data: The dictionary containing the training data.
            val_data: The dictionary containing the validation data.
            test_data: The dictionary containing the testing data.
            predict_data: The dictionary containing the data to use when predicting.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            max_source_length: Max length of the sequence to be considered during tokenization.
            max_target_length: Max length of each answer to be produced.
            padding: Padding type during tokenization.
            question_column_name: The key in the JSON file to recognize the question field.
            context_column_name: The key in the JSON file to recognize the context field.
            answer_column_name: The key in the JSON file to recognize the answer field.
            doc_stride: The stride amount to be taken when splitting up a long document into chunks.

        Returns:
            The constructed data module.
        """

        ds_kw = dict(
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
            question_column_name=question_column_name,
            context_column_name=context_column_name,
            answer_column_name=answer_column_name,
            doc_stride=doc_stride,
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_data, transform=train_transform, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_data, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_data, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_data, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )
