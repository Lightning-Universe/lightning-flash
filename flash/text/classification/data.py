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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import torch
from pandas.core.frame import DataFrame
from torch import Tensor
from torch.utils.data.sampler import Sampler

import flash
from flash.core.data.auto_dataset import AutoDataset
from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_module import DataModule
from flash.core.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources, LabelsState
from flash.core.data.process import Deserializer, Postprocess, Preprocess
from flash.core.integrations.labelstudio.data_source import LabelStudioTextClassificationDataSource
from flash.core.utilities.imports import _TEXT_AVAILABLE, requires

if _TEXT_AVAILABLE:
    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer, default_data_collator
    from transformers.modeling_outputs import SequenceClassifierOutput


class TextDeserializer(Deserializer):
    @requires("text")
    def __init__(self, backbone: str, max_length: int, use_fast: bool = True, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=use_fast, **kwargs)
        self.max_length = max_length

    def deserialize(self, text: str) -> Tensor:
        return self.tokenizer(text, max_length=self.max_length, truncation=True, padding="max_length")

    @property
    def example_input(self) -> str:
        return "An example input"

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("tokenizer")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True)


class TextDataSource(DataSource):
    @requires("text")
    def __init__(self, backbone: str, max_length: int = 128):
        super().__init__()

        self.backbone = backbone
        self.tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
        self.max_length = max_length

    def _tokenize_fn(
        self,
        ex: Union[Dict[str, str], str],
        input: Optional[str] = None,
    ) -> Callable:
        """This function is used to tokenize sentences using the provided tokenizer."""
        return self.tokenizer(ex[input], max_length=self.max_length, truncation=True, padding="max_length")

    @staticmethod
    def _transform_label(label_to_class_mapping: Dict[str, int], target: str, ex: Dict[str, Union[int, str]]):
        ex[target] = label_to_class_mapping[ex[target]]
        return ex

    @staticmethod
    def _multilabel_target(targets: List[str], element: Dict[str, Any]) -> Dict[str, Any]:
        targets = [element.pop(target) for target in targets]
        element[DefaultDataKeys.TARGET] = targets
        return element

    def _to_hf_dataset(self, data) -> Sequence[Mapping[str, Any]]:
        """account for flash CI testing context."""
        hf_dataset, *other = self.to_hf_dataset(data)

        if flash._IS_TESTING and not torch.cuda.is_available():
            # NOTE: must subset in this way to return a Dataset
            hf_dataset = hf_dataset.select(range(20))

        return (hf_dataset, *other)

    def load_data(
        self,
        data: Tuple[str, Union[str, List[str]], Union[str, List[str]]],
        dataset: Optional[Any] = None,
    ) -> Sequence[Mapping[str, Any]]:
        """Loads data into HuggingFace datasets.Dataset."""

        hf_dataset, input, *other = self._to_hf_dataset(data)

        if not self.predicting:
            target: Union[str, List[str]] = other.pop()
            if isinstance(target, List):
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
                hf_dataset = hf_dataset.rename_column(target, DefaultDataKeys.TARGET)

        # remove extra columns
        extra_columns = set(hf_dataset.column_names) - {input, DefaultDataKeys.TARGET}
        hf_dataset = hf_dataset.remove_columns(extra_columns)

        # tokenize
        hf_dataset = hf_dataset.map(partial(self._tokenize_fn, input=input), batched=True, remove_columns=[input])

        # set format
        hf_dataset.set_format("torch")

        return hf_dataset

    def predict_load_data(self, data: Any, dataset: AutoDataset):
        return self.load_data(data, dataset)

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("tokenizer")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True)


class TextCSVDataSource(TextDataSource):
    def to_hf_dataset(self, data: Tuple[str, str, str]) -> Tuple[Sequence[Mapping[str, Any]], str, str]:
        file, *other = data
        dataset_dict = load_dataset("csv", data_files={"train": str(file)})
        return (dataset_dict["train"], *other)


class TextJSONDataSource(TextDataSource):
    def to_hf_dataset(self, data: Tuple[str, str, str, str]) -> Tuple[Sequence[Mapping[str, Any]], str, str]:
        file, *other, field = data
        dataset_dict = load_dataset("json", data_files={"train": str(file)}, field=field)
        return (dataset_dict["train"], *other)


class TextDataFrameDataSource(TextDataSource):
    def to_hf_dataset(self, data: Tuple[DataFrame, str, str]) -> Tuple[Sequence[Mapping[str, Any]], str, str]:
        df, *other = data
        hf_dataset = Dataset.from_pandas(df)
        return (hf_dataset, *other)


class TextParquetDataSource(TextDataSource):
    def to_hf_dataset(self, data: Tuple[str, str, str]) -> Tuple[Sequence[Mapping[str, Any]], str, str]:
        file, *other = data
        hf_dataset = Dataset.from_parquet(str(file))
        return (hf_dataset, *other)


class TextHuggingFaceDatasetDataSource(TextDataSource):
    def to_hf_dataset(self, data: Tuple[str, str, str]) -> Tuple[Sequence[Mapping[str, Any]], str, str]:
        hf_dataset, *other = data
        return (hf_dataset, *other)


class TextListDataSource(TextDataSource):
    def to_hf_dataset(
        self, data: Union[Tuple[List[str], List[str]], List[str]]
    ) -> Tuple[Sequence[Mapping[str, Any]], Optional[List[str]]]:

        if isinstance(data, tuple):
            input_list, target_list = data
            # NOTE: here we already deal with multilabels
            # NOTE: here we already rename to correct column names
            hf_dataset = Dataset.from_dict({DefaultDataKeys.INPUT: input_list, DefaultDataKeys.TARGET: target_list})
            return hf_dataset, target_list

        # predicting
        hf_dataset = Dataset.from_dict({DefaultDataKeys.INPUT: data})

        return (hf_dataset,)

    def load_data(
        self,
        data: Tuple[List[str], Union[List[Any], List[List[Any]]]],
        dataset: Optional[Any] = None,
    ) -> Sequence[Mapping[str, Any]]:

        hf_dataset, *other = self._to_hf_dataset(data)

        if not self.predicting:
            target_list = other.pop()
            if isinstance(target_list[0], List):
                # multi-target_list
                dataset.multi_label = True
                dataset.num_classes = len(target_list[0])
                self.set_state(LabelsState(target_list))
            else:
                dataset.multi_label = False
                if self.training:
                    labels = list(sorted(list(set(hf_dataset[DefaultDataKeys.TARGET]))))
                    dataset.num_classes = len(labels)
                    self.set_state(LabelsState(labels))

                labels = self.get_state(LabelsState)

                # convert labels to ids
                if labels is not None:
                    labels = labels.labels
                    label_to_class_mapping = {v: k for k, v in enumerate(labels)}
                    # happens in-place and keeps the target column name
                    hf_dataset = hf_dataset.map(
                        partial(self._transform_label, label_to_class_mapping, DefaultDataKeys.TARGET)
                    )

        # tokenize
        hf_dataset = hf_dataset.map(partial(self._tokenize_fn, input=DefaultDataKeys.INPUT), batched=True)

        # set format
        hf_dataset = hf_dataset.remove_columns([DefaultDataKeys.INPUT])  # just leave the numerical columns
        hf_dataset.set_format("torch")

        return hf_dataset


class TextClassificationPreprocess(Preprocess):
    @requires("text")
    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        backbone: str = "prajjwal1/bert-tiny",
        max_length: int = 128,
    ):
        self.backbone = backbone
        self.max_length = max_length

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources={
                DefaultDataSources.CSV: TextCSVDataSource(self.backbone, max_length=max_length),
                DefaultDataSources.JSON: TextJSONDataSource(self.backbone, max_length=max_length),
                DefaultDataSources.PARQUET: TextParquetDataSource(self.backbone, max_length=max_length),
                DefaultDataSources.HUGGINGFACE_DATASET: TextHuggingFaceDatasetDataSource(
                    self.backbone, max_length=max_length
                ),
                DefaultDataSources.DATAFRAME: TextDataFrameDataSource(self.backbone, max_length=max_length),
                DefaultDataSources.LISTS: TextListDataSource(self.backbone, max_length=max_length),
                DefaultDataSources.LABELSTUDIO: LabelStudioTextClassificationDataSource(
                    backbone=self.backbone, max_length=max_length
                ),
            },
            default_data_source=DefaultDataSources.LISTS,
            deserializer=TextDeserializer(backbone, max_length),
        )

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            **self.transforms,
            "backbone": self.backbone,
            "max_length": self.max_length,
        }

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


class TextClassificationPostprocess(Postprocess):
    def per_batch_transform(self, batch: Any) -> Any:
        if isinstance(batch, SequenceClassifierOutput):
            batch = batch.logits
        return super().per_batch_transform(batch)


class TextClassificationData(DataModule):
    """Data Module for text classification tasks."""

    preprocess_cls = TextClassificationPreprocess
    postprocess_cls = TextClassificationPostprocess

    @property
    def backbone(self) -> Optional[str]:
        return getattr(self.preprocess, "backbone", None)

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
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: int = 0,
        sampler: Optional[Type[Sampler]] = None,
        **preprocess_kwargs: Any,
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
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            data_fetcher: The :class:`~flash.core.data.callback.BaseDataFetcher` to pass to the
                :class:`~flash.core.data.data_module.DataModule`.
            preprocess: The :class:`~flash.core.data.data.Preprocess` to pass to the
                :class:`~flash.core.data.data_module.DataModule`. If ``None``, ``cls.preprocess_cls``
                will be constructed and used.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            batch_size: The ``batch_size`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            num_workers: The ``num_workers`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            sampler: The ``sampler`` to use for the ``train_dataloader``.
            preprocess_kwargs: Additional keyword arguments to use when constructing the preprocess. Will only be used
                if ``preprocess = None``.

        Returns:
            The constructed data module.
        """
        return cls.from_data_source(
            DefaultDataSources.DATAFRAME,
            (train_data_frame, input_field, target_fields),
            (val_data_frame, input_field, target_fields),
            (test_data_frame, input_field, target_fields),
            (predict_data_frame, input_field, target_fields),
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            **preprocess_kwargs,
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
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: int = 0,
        sampler: Optional[Type[Sampler]] = None,
        **preprocess_kwargs: Any,
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
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            data_fetcher: The :class:`~flash.core.data.callback.BaseDataFetcher` to pass to the
                :class:`~flash.core.data.data_module.DataModule`.
            preprocess: The :class:`~flash.core.data.data.Preprocess` to pass to the
                :class:`~flash.core.data.data_module.DataModule`. If ``None``, ``cls.preprocess_cls``
                will be constructed and used.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            batch_size: The ``batch_size`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            num_workers: The ``num_workers`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            sampler: The ``sampler`` to use for the ``train_dataloader``.
            preprocess_kwargs: Additional keyword arguments to use when constructing the preprocess. Will only be used
                if ``preprocess = None``.

        Returns:
            The constructed data module.
        """
        return cls.from_data_source(
            DefaultDataSources.LISTS,
            (train_data, train_targets),
            (val_data, val_targets),
            (test_data, test_targets),
            predict_data,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            **preprocess_kwargs,
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
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: int = 0,
        sampler: Optional[Type[Sampler]] = None,
        **preprocess_kwargs: Any,
    ) -> "DataModule":
        """Creates a :class:`~flash.core.data.data_module.DataModule` object from the given PARQUET files using the
        :class:`~flash.core.data.data_source.DataSource`
        of name :attr:`~flash.core.data.data_source.DefaultDataSources.PARQUET`
        from the passed or constructed :class:`~flash.core.data.process.Preprocess`.

        Args:
            input_fields: The field or fields (columns) in the PARQUET file to use for the input.
            target_fields: The field or fields (columns) in the PARQUET file to use for the target.
            train_file: The PARQUET file containing the training data.
            val_file: The PARQUET file containing the validation data.
            test_file: The PARQUET file containing the testing data.
            predict_file: The PARQUET file containing the data to use when predicting.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            data_fetcher: The :class:`~flash.core.data.callback.BaseDataFetcher` to pass to the
                :class:`~flash.core.data.data_module.DataModule`.
            preprocess: The :class:`~flash.core.data.data.Preprocess` to pass to the
                :class:`~flash.core.data.data_module.DataModule`. If ``None``, ``cls.preprocess_cls``
                will be constructed and used.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            batch_size: The ``batch_size`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            num_workers: The ``num_workers`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            sampler: The ``sampler`` to use for the ``train_dataloader``.
            preprocess_kwargs: Additional keyword arguments to use when constructing the preprocess. Will only be used
                if ``preprocess = None``.

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
        return cls.from_data_source(
            DefaultDataSources.PARQUET,
            (train_file, input_field, target_fields),
            (val_file, input_field, target_fields),
            (test_file, input_field, target_fields),
            (predict_file, input_field, target_fields),
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            **preprocess_kwargs,
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
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: int = 0,
        sampler: Optional[Type[Sampler]] = None,
        **preprocess_kwargs: Any,
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
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.process.Preprocess` hook names to callable transforms.
            data_fetcher: The :class:`~flash.core.data.callback.BaseDataFetcher` to pass to the
                :class:`~flash.core.data.data_module.DataModule`.
            preprocess: The :class:`~flash.core.data.data.Preprocess` to pass to the
                :class:`~flash.core.data.data_module.DataModule`. If ``None``, ``cls.preprocess_cls``
                will be constructed and used.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            batch_size: The ``batch_size`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            num_workers: The ``num_workers`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            sampler: The ``sampler`` to use for the ``train_dataloader``.
            preprocess_kwargs: Additional keyword arguments to use when constructing the preprocess. Will only be used
                if ``preprocess = None``.

        Returns:
            The constructed data module.
        """
        return cls.from_data_source(
            DefaultDataSources.HUGGINGFACE_DATASET,
            (train_hf_dataset, input_field, target_fields),
            (val_hf_dataset, input_field, target_fields),
            (test_hf_dataset, input_field, target_fields),
            (predict_hf_dataset, input_field, target_fields),
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            **preprocess_kwargs,
        )
