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
import os
from functools import partial
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import torch
from pandas.core.frame import DataFrame
from torch import Tensor
from torch.utils.data.sampler import Sampler

import flash
from flash.core.data.auto_dataset import AutoDataset
from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_module import DataModule
from flash.core.data.io.input import DataKeys, Input, InputFormat, LabelsState
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.io.output_transform import OutputTransform
from flash.core.data.process import Deserializer
from flash.core.integrations.labelstudio.input import LabelStudioTextClassificationInput
from flash.core.utilities.imports import _TEXT_AVAILABLE, requires
from flash.text.classification.tokenizers.base import BaseTokenizer

if _TEXT_AVAILABLE:
    from datasets import Dataset, load_dataset
    from transformers import default_data_collator
    from transformers.modeling_outputs import SequenceClassifierOutput

    from flash.text.classification.tokenizers import TEXT_CLASSIFIER_TOKENIZERS


class TextDeserializer(Deserializer):
    @requires("text")
    def __init__(self, tokenizer: BaseTokenizer):
        super().__init__()

        self.tokenizer = tokenizer

    def deserialize(self, text: str) -> Tensor:
        return self.tokenizer(text)

    @property
    def example_input(self) -> str:
        return "An example input"


class TextInput(Input):
    @requires("text")
    def __init__(self, tokenizer: BaseTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def _tokenize_fn(
        self,
        ex: Union[Dict[str, str], str],
        input: Optional[str] = None,
    ) -> Callable:
        """This function is used to tokenize sentences using the provided tokenizer."""
        return self.tokenizer(ex[input])

    @staticmethod
    def _transform_label(label_to_class_mapping: Dict[str, int], target: str, ex: Dict[str, Union[int, str]]):
        ex[target] = label_to_class_mapping[ex[target]]
        return ex

    @staticmethod
    def _multilabel_target(targets: List[str], element: Dict[str, Any]) -> Dict[str, Any]:
        targets = [element.pop(target) for target in targets]
        element[DataKeys.TARGET] = targets
        return element

    def _to_hf_dataset(self, data) -> Sequence[Mapping[str, Any]]:
        """account for flash CI testing context."""
        hf_dataset, *other = self.to_hf_dataset(data)

        if flash._IS_TESTING and not torch.cuda.is_available():
            # NOTE: must subset in this way to return a Dataset
            hf_dataset = hf_dataset.select(range(20))

        return (hf_dataset, *other)

    def _encode_target(self, hf_dataset, dataset, target) -> Sequence[Mapping[str, Any]]:
        if isinstance(target, list):
            # multi-target
            dataset.multi_label = True
            hf_dataset = hf_dataset.map(partial(self._multilabel_target, target))  # NOTE: renames target column
            dataset.num_classes = len(target)
            self.set_state(LabelsState(target))
        else:
            dataset.multi_label = False
            if self.training:
                labels = list(sorted(list(set(hf_dataset[target]))))
                dataset.num_classes = len(labels)
                self.set_state(LabelsState(labels))

            labels = self.get_state(LabelsState)

            # convert labels to ids (note: the target column get overwritten)
            if labels is not None:
                labels = labels.labels
                label_to_class_mapping = {v: k for k, v in enumerate(labels)}
                hf_dataset = hf_dataset.map(partial(self._transform_label, label_to_class_mapping, target))

            # rename label column
            hf_dataset = hf_dataset.rename_column(target, DataKeys.TARGET)

        return hf_dataset

    def _encode_input(self, hf_dataset, input) -> Sequence[Mapping[str, Any]]:
        # tokenize
        if not self.tokenizer.is_fitted:
            self.tokenizer.fit(hf_dataset, input=input)
        hf_dataset = hf_dataset.map(partial(self._tokenize_fn, input=input), batched=True)
        hf_dataset = hf_dataset.remove_columns([input])  # just leave the numerical columns

        return hf_dataset

    def load_data(
        self,
        data: Tuple[str, Union[str, List[str]], Union[str, List[str]]],
        dataset: Optional[Any] = None,
    ) -> Sequence[Mapping[str, Any]]:
        """Loads data into HuggingFace datasets.Dataset."""

        hf_dataset, input, *other = self._to_hf_dataset(data)

        if not self.predicting:
            target: Union[str, List[str]] = other.pop()
            hf_dataset = self._encode_target(hf_dataset, dataset, target)

        hf_dataset = self._encode_input(hf_dataset, input)

        return hf_dataset

    def predict_load_data(self, data: Any, dataset: AutoDataset):
        return self.load_data(data, dataset)


class TextCSVInput(TextInput):
    def to_hf_dataset(self, data: Tuple[str, str, str]) -> Tuple[Sequence[Mapping[str, Any]], str, Optional[List[str]]]:
        file, input, *other = data
        dataset_dict = load_dataset("csv", data_files={"train": str(file)})
        return (dataset_dict["train"], input, *other)


class TextJSONInput(TextInput):
    def to_hf_dataset(
        self, data: Tuple[str, str, str, str]
    ) -> Tuple[Sequence[Mapping[str, Any]], str, Optional[List[str]]]:
        file, input, *other, field = data
        dataset_dict = load_dataset("json", data_files={"train": str(file)}, field=field)
        return (dataset_dict["train"], input, *other)


class TextDataFrameInput(TextInput):
    def to_hf_dataset(
        self, data: Tuple[DataFrame, str, str]
    ) -> Tuple[Sequence[Mapping[str, Any]], str, Optional[List[str]]]:
        df, input, *other = data
        hf_dataset = Dataset.from_pandas(df)
        return (hf_dataset, input, *other)


class TextParquetInput(TextInput):
    def to_hf_dataset(self, data: Tuple[str, str, str]) -> Tuple[Sequence[Mapping[str, Any]], str, Optional[List[str]]]:
        file, input, *other = data
        hf_dataset = Dataset.from_parquet(str(file))
        return (hf_dataset, input, *other)


class TextHuggingFaceDatasetInput(TextInput):
    def to_hf_dataset(self, data: Tuple[str, str, str]) -> Tuple[Sequence[Mapping[str, Any]], str, Optional[List[str]]]:
        hf_dataset, input, *other = data
        return (hf_dataset, input, *other)


class TextListInput(TextInput):
    def to_hf_dataset(
        self, data: Union[Tuple[List[str], List[str]], List[str]]
    ) -> Tuple[Sequence[Mapping[str, Any]], str, Optional[List[str]]]:

        if isinstance(data, tuple):
            input_list, target_list = data
            # NOTE: here we already deal with multilabels
            # NOTE: here we already rename to correct column names
            hf_dataset = Dataset.from_dict({DataKeys.INPUT: input_list, DataKeys.TARGET: target_list})
            return hf_dataset, DataKeys.INPUT, target_list

        # predicting
        hf_dataset = Dataset.from_dict({DataKeys.INPUT: data})

        return (hf_dataset, DataKeys.INPUT)

    def _encode_target(self, hf_dataset, dataset, target) -> Sequence[Mapping[str, Any]]:
        if isinstance(target[0], List):
            # multi-target
            dataset.multi_label = True
            dataset.num_classes = len(target[0])
            self.set_state(LabelsState(target))
        else:
            dataset.multi_label = False
            if self.training:
                labels = list(sorted(list(set(hf_dataset[DataKeys.TARGET]))))
                dataset.num_classes = len(labels)
                self.set_state(LabelsState(labels))

            labels = self.get_state(LabelsState)

            # convert labels to ids
            if labels is not None:
                labels = labels.labels
                label_to_class_mapping = {v: k for k, v in enumerate(labels)}
                # happens in-place and keeps the target column name
                hf_dataset = hf_dataset.map(
                    partial(self._transform_label, label_to_class_mapping, DataKeys.TARGET)
                )

        return hf_dataset


class TextClassificationInputTransform(InputTransform):
    @requires("text")
    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        backbone: Union[str, Tuple[BaseTokenizer, int]] = "prajjwal1/bert-tiny",
        pretrained: Optional[bool] = True,
        **backbone_kwargs: Optional[Dict[str, Any]],
    ):
        if isinstance(backbone, tuple):
            self.tokenizer, self.vocab_size = backbone
            self.backbone = self.tokenizer.backbone
        else:
            self.backbone = backbone
            self.tokenizer, self.vocab_size = TEXT_CLASSIFIER_TOKENIZERS.get(backbone)(
                pretrained=pretrained, **backbone_kwargs
            )

        os.environ["TOKENIZERS_PARALLELISM"] = "true"  # TODO: do we really need this?

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources={
                InputFormat.CSV: TextCSVInput(self.tokenizer),
                InputFormat.JSON: TextJSONInput(self.tokenizer),
                InputFormat.PARQUET: TextParquetInput(self.tokenizer),
                InputFormat.HUGGINGFACE_DATASET: TextHuggingFaceDatasetInput(self.tokenizer),
                InputFormat.DATAFRAME: TextDataFrameInput(self.tokenizer),
                InputFormat.LISTS: TextListInput(self.tokenizer),
                InputFormat.LABELSTUDIO: LabelStudioTextClassificationInput(self.tokenizer),
            },
            default_data_source=InputFormat.LISTS,
            deserializer=TextDeserializer(self.tokenizer),
        )

    def get_state_dict(self) -> Dict[str, Any]:
        return {**self.transforms}

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool):
        return cls(**state_dict)

    def per_batch_transform(self, batch: Any) -> Any:
        if "labels" not in batch:
            # todo: understand why an extra dimension has been added.
            if batch["input_ids"].dim() == 3:
                batch["input_ids"] = batch["input_ids"].squeeze(0)
        return batch

    def collate(self, samples: Any) -> Tensor:
        """Override to convert a set of samples to a batch."""
        if isinstance(samples, dict):
            samples = [samples]
        return default_data_collator(samples)


class TextClassificationOutputTransform(OutputTransform):
    def per_batch_transform(self, batch: Any) -> Any:
        if isinstance(batch, SequenceClassifierOutput):
            batch = batch.logits
        return super().per_batch_transform(batch)


class TextClassificationData(DataModule):
    """Data Module for text classification tasks."""

    input_transform_cls = TextClassificationInputTransform
    output_transform_cls = TextClassificationOutputTransform

    @property
    def backbone(self) -> Optional[str]:
        return getattr(self.input_transform, "backbone", None)

    @classmethod
    def from_data_frame(
        cls,
        input_field: str,
        target_fields: Union[str, Sequence[str]],
        train_data_frame: Optional[DataFrame] = None,
        val_data_frame: Optional[DataFrame] = None,
        test_data_frame: Optional[DataFrame] = None,
        predict_data_frame: Optional[DataFrame] = None,
        train_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        val_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        test_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        input_transform: Optional[InputTransform] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: int = 0,
        sampler: Optional[Type[Sampler]] = None,
        **input_transform_kwargs: Any,
    ) -> "DataModule":
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
            data_fetcher: The :class:`~flash.core.data.callback.BaseDataFetcher` to pass to the
                :class:`~flash.core.data.data_module.DataModule`.
            input_transform: The :class:`~flash.core.data.data.InputTransform` to pass to the
                :class:`~flash.core.data.data_module.DataModule`. If ``None``, ``cls.input_transform_cls``
                will be constructed and used.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            batch_size: The ``batch_size`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            num_workers: The ``num_workers`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            sampler: The ``sampler`` to use for the ``train_dataloader``.
            input_transform_kwargs: Additional keyword arguments to use when constructing the input_transform.
                Will only be used if ``input_transform = None``.

        Returns:
            The constructed data module.
        """
        return cls.from_input(
            InputFormat.DATAFRAME,
            (train_data_frame, input_field, target_fields),
            (val_data_frame, input_field, target_fields),
            (test_data_frame, input_field, target_fields),
            (predict_data_frame, input_field, target_fields),
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            input_transform=input_transform,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            **input_transform_kwargs,
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
        data_fetcher: Optional[BaseDataFetcher] = None,
        input_transform: Optional[InputTransform] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: int = 0,
        sampler: Optional[Type[Sampler]] = None,
        **input_transform_kwargs: Any,
    ) -> "DataModule":
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
            data_fetcher: The :class:`~flash.core.data.callback.BaseDataFetcher` to pass to the
                :class:`~flash.core.data.data_module.DataModule`.
            input_transform: The :class:`~flash.core.data.data.InputTransform` to pass to the
                :class:`~flash.core.data.data_module.DataModule`. If ``None``, ``cls.input_transform_cls``
                will be constructed and used.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            batch_size: The ``batch_size`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            num_workers: The ``num_workers`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            sampler: The ``sampler`` to use for the ``train_dataloader``.
            input_transform_kwargs: Additional keyword arguments to use when constructing the input_transform.
                Will only be used if ``input_transform = None``.

        Returns:
            The constructed data module.
        """
        return cls.from_input(
            InputFormat.LISTS,
            (train_data, train_targets),
            (val_data, val_targets),
            (test_data, test_targets),
            predict_data,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            input_transform=input_transform,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            **input_transform_kwargs,
        )

    @classmethod
    def from_parquet(
        cls,
        input_field: str,
        target_fields: Optional[Union[str, Sequence[str]]] = None,
        train_file: Optional[str] = None,
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        predict_file: Optional[str] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        input_transform: Optional[InputTransform] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: int = 0,
        sampler: Optional[Type[Sampler]] = None,
        **input_transform_kwargs: Any,
    ) -> "DataModule":
        """Creates a :class:`~flash.core.data.data_module.DataModule` object from the given PARQUET files using the
        :class:`~flash.core.data.io.input.Input`
        of name :attr:`~flash.core.data.io.input.InputFormat.PARQUET`
        from the passed or constructed :class:`~flash.core.data.io.input_transform.InputTransform`.

        Args:
            input_fields: The field or fields (columns) in the PARQUET file to use for the input.
            target_fields: The field or fields (columns) in the PARQUET file to use for the target.
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
            data_fetcher: The :class:`~flash.core.data.callback.BaseDataFetcher` to pass to the
                :class:`~flash.core.data.data_module.DataModule`.
            input_transform: The :class:`~flash.core.data.data.InputTransform` to pass to the
                :class:`~flash.core.data.data_module.DataModule`. If ``None``, ``cls.input_transform_cls``
                will be constructed and used.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            batch_size: The ``batch_size`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            num_workers: The ``num_workers`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            sampler: The ``sampler`` to use for the ``train_dataloader``.
            input_transform_kwargs: Additional keyword arguments to use when constructing the input_transform.
                Will only be used if ``input_transform = None``.

        Returns:
            The constructed data module.

        Examples::

            data_module = DataModule.from_parquet(
                "input",
                "target",
                train_file="train_data.parquet",
                train_transform={
                    "to_tensor_transform": torch.as_tensor,
                },
            )
        """
        return cls.from_input(
            InputFormat.PARQUET,
            (train_file, input_field, target_fields),
            (val_file, input_field, target_fields),
            (test_file, input_field, target_fields),
            (predict_file, input_field, target_fields),
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            input_transform=input_transform,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            **input_transform_kwargs,
        )

    @classmethod
    def from_hf_datasets(
        cls,
        input_field: str,
        target_fields: Union[str, Sequence[str]],
        train_hf_dataset: Optional[Sequence[Mapping[str, Any]]] = None,
        val_hf_dataset: Optional[Sequence[Mapping[str, Any]]] = None,
        test_hf_dataset: Optional[Sequence[Mapping[str, Any]]] = None,
        predict_hf_dataset: Optional[Sequence[Mapping[str, Any]]] = None,
        train_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        val_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        test_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        input_transform: Optional[InputTransform] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: int = 0,
        sampler: Optional[Type[Sampler]] = None,
        **input_transform_kwargs: Any,
    ) -> "DataModule":
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
            data_fetcher: The :class:`~flash.core.data.callback.BaseDataFetcher` to pass to the
                :class:`~flash.core.data.data_module.DataModule`.
            input_transform: The :class:`~flash.core.data.data.InputTransform` to pass to the
                :class:`~flash.core.data.data_module.DataModule`. If ``None``, ``cls.input_transform_cls``
                will be constructed and used.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            batch_size: The ``batch_size`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            num_workers: The ``num_workers`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            sampler: The ``sampler`` to use for the ``train_dataloader``.
            input_transform_kwargs: Additional keyword arguments to use when constructing the input_transform.
                Will only be used if ``input_transform = None``.

        Returns:
            The constructed data module.
        """
        return cls.from_input(
            InputFormat.HUGGINGFACE_DATASET,
            (train_hf_dataset, input_field, target_fields),
            (val_hf_dataset, input_field, target_fields),
            (test_hf_dataset, input_field, target_fields),
            (predict_hf_dataset, input_field, target_fields),
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            input_transform=input_transform,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            **input_transform_kwargs,
        )
