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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TypeVar, Union, Type

import datasets
from pandas.core.frame import DataFrame
import torch
from pytorch_lightning.trainer.states import RunningStage
from torch import Tensor
from torch.utils.data.sampler import Sampler

from flash.core.data.auto_dataset import AutoDataset, IterableAutoDataset
from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_module import DataModule
from flash.core.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources, LabelsState
from flash.core.data.process import Deserializer, Postprocess, Preprocess, Serializer
from flash.core.integrations.labelstudio.data_source import LabelStudioTextClassificationDataSource
from flash.core.utilities.imports import _TEXT_AVAILABLE, requires
from flash.text.classification.tokenizers.base import BaseTokenizer

if _TEXT_AVAILABLE:
    from datasets import Dataset, load_dataset

    from flash.text.classification.tokenizers import TEXT_CLASSIFIER_TOKENIZERS


DATA_TYPE = TypeVar("DATA_TYPE")


class TextDeserializer(Deserializer):
    @requires("text")
    def __init__(self, backbone: Union[str, BaseTokenizer], **backbone_kwargs):
        super().__init__()
        
        if isinstance(backbone, str):
            self.tokenizer, _ = TEXT_CLASSIFIER_TOKENIZERS.get(backbone)(**backbone_kwargs)
            self.backbone = backbone
        else:
            self.tokenizer = backbone
            self.backbone = self.tokenizer.backbone

    def deserialize(self, text: Union[str, List[str]]) -> Tensor:
        return self.tokenizer(text, return_tensors="pt")

    @property
    def example_input(self) -> str:
        return "An example input"


class TextSerializer(Serializer):
    @requires("text")
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def serialize(self, token_ids: Union[int, List[int]]) -> str:
        return self.tokenizer.decode(token_ids)


class TextDataSource(DataSource):
    @requires("text")
    def __init__(self, tokenizer, vocab_size: int):
        super().__init__()

        self.tokenizer = tokenizer
        self.vocab_size = vocab_size

    @staticmethod
    def _transform_label(label_to_class_mapping: Dict[str, int], target: str, ex: Dict[str, Union[int, str]]):
        ex[target] = label_to_class_mapping[ex[target]]
        return ex

    @staticmethod
    def _multilabel_target(targets, element):
        targets = [element.pop(target) for target in targets]
        element[DefaultDataKeys.TARGET] = targets
        return element

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("tokenizer")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer, self.vocab_size = TEXT_CLASSIFIER_TOKENIZERS.get(self.backbone)(**self.backbone_kwargs)

    def generate_dataset(
        self,
        data: Optional[DATA_TYPE],
        running_stage: RunningStage,
    ) -> Optional[Union[AutoDataset, IterableAutoDataset]]:
        """Generate a single dataset with the given input to
        :meth:`~flash.core.data.data_source.DataSource.load_data` for the given ``running_stage``.
        Args:
            data: The input to :meth:`~flash.core.data.data_source.DataSource.load_data` to use to create the dataset.
            running_stage: The running_stage for this dataset.
        Returns:
            The constructed :class:`~flash.core.data.auto_dataset.BaseAutoDataset`.
        """

        dataset: Union[AutoDataset, IterableAutoDataset] = super().generate_dataset(data, running_stage)

        # predict might not be present
        if not dataset:
            return

        # decide whether to fit tokenizer
        if running_stage == RunningStage.TRAINING and not self.tokenizer._is_fit:
            batch_iterator = self.tokenizer._batch_iterator(dataset)
            self.tokenizer.fit(batch_iterator)  # TODO: save state to disk
            print(
                f"Tokenizer fit with `vocab_size={self.tokenizer.vocab_size}`, `max_length={self.tokenizer.max_length}`, `batch_size={self.tokenizer.batch_size}`"
            )

        return dataset

    def load_data(
        self,
        data: Tuple[str, Union[str, List[str]], Union[str, List[str]]],
        dataset: Optional[Any] = None,
    ) -> Dataset:
        """Loads data into HuggingFace datasets.Dataset"""

        hf_dataset, input, target = self.to_hf_dataset(data)
        
        if not self.predicting:
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

                # convert labels to ids
                if labels is not None:
                    labels = labels.labels
                    label_to_class_mapping = {v: k for k, v in enumerate(labels)}
                    hf_dataset = hf_dataset.map(partial(self._transform_label, label_to_class_mapping, target))

                # rename label column
                hf_dataset = hf_dataset.rename_column(target, DefaultDataKeys.TARGET)

        # rename input column
        hf_dataset = hf_dataset.rename_column(input, DefaultDataKeys.INPUT)

        return hf_dataset

    def predict_load_data(self, data: Any, dataset: AutoDataset):
        return self.load_data(data, dataset)


class TextCSVDataSource(TextDataSource):
    def to_hf_dataset(self, data: Tuple[str, str, str]) -> Tuple[Dataset, str, str]:
        file, input, target = data
        dataset_dict = load_dataset("csv", data_files={"train": str(file)})
        return dataset_dict["train"], input, target


class TextJSONDataSource(TextDataSource):
    def to_hf_dataset(self, data: Tuple[str, str, str, str]) -> Tuple[Dataset, str, str]:
        file, input, target, field = data
        dataset_dict = load_dataset("json", data_files={"train": str(file)}, field=field)
        return dataset_dict["train"], input, target


class TextDataFrameDataSource(TextDataSource):    
    def to_hf_dataset(self, data: Tuple[DataFrame, str, str]) -> Tuple[Dataset, str, str]:
        df, input, target = data
        hf_dataset = Dataset.from_pandas(df)
        return hf_dataset, input, target


class TextParquetDataSource(TextDataSource):
    def to_hf_dataset(self, data: Tuple[str, str, str]) -> Tuple[Dataset, str, str]:
        file, input, target = data
        hf_dataset = Dataset.from_parquet(file)
        return hf_dataset, input, target


class TextHuggingFaceDatasetDataSource(TextDataSource):
    def to_hf_dataset(self, data: Tuple[str, str, str]) -> Tuple[Dataset, str, str]:
        hf_dataset, input, target = data
        return hf_dataset, input, target


class TextListDataSource(TextDataSource):    
    def to_hf_dataset(self, data: Tuple[List[str], List[str]]) -> Tuple[Dataset, List[str], List[str]]:
        input, target = data
        if target:
            hf_dataset = Dataset.from_dict({"input": input, "labels": target})
        else:
            # predicting
            hf_dataset = Dataset.from_dict({"input": input})
        return hf_dataset, input, target

    def load_data(
        self,
        data: Tuple[List[str], Union[List[Any], List[List[Any]]]],
        dataset: Optional[Any] = None,
    ) -> Union[Sequence[Mapping[str, Any]]]:
        
        hf_dataset, input, target = self.to_hf_dataset(data)

        if not self.predicting:
            if isinstance(target[0], List):
                # multi-target
                dataset.multi_label = True
                dataset.num_classes = len(target[0])
                self.set_state(LabelsState(target))
            else:
                dataset.multi_label = False
                if self.training:
                    labels = list(sorted(list(set(hf_dataset["labels"]))))
                    dataset.num_classes = len(labels)
                    self.set_state(LabelsState(labels))

                labels = self.get_state(LabelsState)

                # convert labels to ids
                if labels is not None:
                    labels = labels.labels
                    label_to_class_mapping = {v: k for k, v in enumerate(labels)}
                    hf_dataset = hf_dataset.map(partial(self._transform_label, label_to_class_mapping, "labels"))

                # rename label column
                hf_dataset = hf_dataset.rename_column("labels", DefaultDataKeys.TARGET)

        return hf_dataset


class TextClassificationPreprocess(Preprocess):
    @requires("text")
    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        backbone: Union[str, Tuple[BaseTokenizer, int]] = "prajjwal1/bert-tiny",
        backbone_kwargs: Optional[Dict[str, Any]] = None,
    ):

        if isinstance(backbone, tuple):
            self.tokenizer, self.vocab_size = backbone
            self.backbone = self.tokenizer.backbone
        else:
            self.backbone = backbone
            self.tokenizer, self.vocab_size = TEXT_CLASSIFIER_TOKENIZERS.get(backbone)(**backbone_kwargs)

        os.environ["TOKENIZERS_PARALLELISM"] = "true"  # TODO: do we really need this?

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources={
                DefaultDataSources.CSV: TextCSVDataSource(self.tokenizer, self.vocab_size),
                DefaultDataSources.JSON: TextJSONDataSource(self.tokenizer, self.vocab_size),
                DefaultDataSources.PARQUET: TextParquetDataSource(self.tokenizer, self.vocab_size),
                DefaultDataSources.DATAFRAME: TextDataFrameDataSource(self.tokenizer, self.vocab_size),
                DefaultDataSources.HUGGINGFACE_DATASET: TextHuggingFaceDatasetDataSource(self.tokenizer, self.vocab_size),
                DefaultDataSources.LISTS: TextListDataSource(self.tokenizer, self.vocab_size),
                DefaultDataSources.LABELSTUDIO: LabelStudioTextClassificationDataSource(self.tokenizer, self.vocab_size),
            },
            default_data_source=DefaultDataSources.LISTS,
            deserializer=TextDeserializer(self.tokenizer),
        )

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            **self.transforms,
            "tokenizer": self.tokenizer,
        }

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool):
        return cls(**state_dict)

    def collate(self, samples: Union[List[Dict[str, Any]], List[str]]) -> Dict[str, Tensor]:
        """Tokenizes inputs and collates."""

        # collate and then tokenize (more efficient)
        collated_batch = {
            DefaultDataKeys.INPUT: self.tokenizer(
                [sample[DefaultDataKeys.INPUT] for sample in samples],
                return_tensors="pt",
            )
        }

        if DefaultDataKeys.TARGET in samples[0]:
            collated_batch[DefaultDataKeys.TARGET] = torch.tensor(
                [sample[DefaultDataKeys.TARGET] for sample in samples],
                dtype=torch.int64,  # like what HuggingFace returns above
            )

        return collated_batch


class TextClassificationPostprocess(Postprocess):
    # TODO: this has to hook into the preprocessor and take the tokenizer
    # as it might be trained
    pass


class TextClassificationData(DataModule):
    """Data Module for text classification tasks."""

    preprocess_cls = TextClassificationPreprocess
    postprocess_cls = TextClassificationPostprocess

    @property
    def backbone(self) -> Optional[str]:
        return getattr(self.preprocess, "backbone", None)

    @property
    def vocab_size(self) -> str:
        return getattr(self.preprocess, "vocab_size")

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
        train_hf_dataset: Optional[datasets.Dataset] = None,
        val_hf_dataset: Optional[datasets.Dataset] = None,
        test_hf_dataset: Optional[datasets.Dataset] = None,
        predict_hf_dataset: Optional[datasets.Dataset] = None,
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
        """Creates a :class:`~flash.text.classification.data.TextClassificationData` object from the given 
        Hugging Face datasets ``Dataset`` objects.

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