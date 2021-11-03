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
from dataclasses import dataclass
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
from flash.core.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources
from flash.core.data.process import Deserializer, Preprocess, ProcessState
from flash.core.utilities.imports import _TEXT_AVAILABLE, requires

if _TEXT_AVAILABLE:
    from datasets import Dataset, load_dataset
    from transformers import default_data_collator

    from flash.text.tokenizers import TEXT_CLASSIFIER_TOKENIZERS
    from flash.text.tokenizers.base import BaseTokenizer


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


@dataclass(unsafe_hash=True, frozen=True)
class TokenizerState(ProcessState):
    """A :class:`~flash.core.data.properties.ProcessState` containing ``tokenizer``."""

    tokenizer: BaseTokenizer


class TextDataSource(DataSource):
    @requires("text")
    def __init__(self, tokenizer: BaseTokenizer):
        super().__init__()
        self.set_state(TokenizerState(tokenizer))

    def _tokenize_fn(
        self,
        ex: Dict[str, str],
        input: Optional[str] = None,
    ) -> Callable:
        """This function is used to tokenize sentences using the provided tokenizer."""
        return self.get_state(TokenizerState).tokenizer(ex[input])

    def encode_input(self, hf_dataset, input) -> Sequence[Mapping[str, Any]]:
        # tokenize
        if not self.get_state(TokenizerState).tokenizer.is_fitted:
            self.get_state(TokenizerState).tokenizer.fit(hf_dataset, input=input)
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

        hf_dataset = self.encode_input(hf_dataset, input)

        if not self.predicting:
            target = other.pop()
            hf_dataset = self.encode_target(hf_dataset, dataset, target)

        return hf_dataset

    def _to_hf_dataset(self, data) -> Sequence[Mapping[str, Any]]:
        """account for flash CI testing context."""
        hf_dataset, *other = self.to_hf_dataset(data)

        if flash._IS_TESTING and not torch.cuda.is_available():
            # NOTE: must subset in this way to return a Dataset
            hf_dataset = hf_dataset.select(range(20))

        return (hf_dataset, *other)

    def predict_load_data(self, data: Any, dataset: AutoDataset):
        return self.load_data(data, dataset)


class TextCSVDataSourceMixin:
    def to_hf_dataset(self, data: Tuple[str, str, str]) -> Tuple[Sequence[Mapping[str, Any]], str, Optional[List[str]]]:
        file, input, *other = data
        dataset_dict = load_dataset("csv", data_files={"train": str(file)})
        return (dataset_dict["train"], input, *other)


class TextJSONDataSourceMixin:
    def to_hf_dataset(
        self, data: Tuple[str, str, str, str]
    ) -> Tuple[Sequence[Mapping[str, Any]], str, Optional[List[str]]]:
        file, input, *other, field = data
        dataset_dict = load_dataset("json", data_files={"train": str(file)}, field=field)
        return (dataset_dict["train"], input, *other)


class TextDataFrameDataSourceMixin:
    def to_hf_dataset(
        self, data: Tuple[DataFrame, str, str]
    ) -> Tuple[Sequence[Mapping[str, Any]], str, Optional[List[str]]]:
        df, input, *other = data
        hf_dataset = Dataset.from_pandas(df)
        return (hf_dataset, input, *other)


class TextParquetDataSourceMixin:
    def to_hf_dataset(self, data: Tuple[str, str, str]) -> Tuple[Sequence[Mapping[str, Any]], str, Optional[List[str]]]:
        file, input, *other = data
        hf_dataset = Dataset.from_parquet(str(file))
        return (hf_dataset, input, *other)


class TextHuggingFaceDatasetDataSourceMixin:
    def to_hf_dataset(self, data: Tuple[str, str, str]) -> Tuple[Sequence[Mapping[str, Any]], str, Optional[List[str]]]:
        hf_dataset, input, *other = data
        return (hf_dataset, input, *other)


class TextListDataSourceMixin:
    def to_hf_dataset(
        self, data: Union[Tuple[List[str], List[str]], List[str]]
    ) -> Tuple[Sequence[Mapping[str, Any]], str, Optional[List[str]]]:

        if isinstance(data, tuple):
            input_list, target_list = data
            # NOTE: here we already deal with multilabels
            # NOTE: here we already rename to correct column names
            hf_dataset = Dataset.from_dict({DefaultDataKeys.INPUT: input_list, DefaultDataKeys.TARGET: target_list})
            return hf_dataset, DefaultDataKeys.INPUT, target_list

        # predicting
        hf_dataset = Dataset.from_dict({DefaultDataKeys.INPUT: data})

        return (hf_dataset, DefaultDataKeys.INPUT)


class TextPreprocess(Preprocess):
    @requires("text")
    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        data_sources: Optional[Dict[str, "DataSource"]] = None,
        backbone: Union[str, Tuple[BaseTokenizer, int]] = None,
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
            data_sources={k: v(self.tokenizer) for k, v in data_sources.items()},
            default_data_source=DefaultDataSources.LISTS,
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


class TextDataModule(DataModule):
    """Data Module for text classification tasks."""

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
