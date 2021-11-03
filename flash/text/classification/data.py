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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from flash.core.data.data_pipeline import Postprocess
from flash.core.data.data_source import DefaultDataKeys, DefaultDataSources, LabelsState
from flash.core.integrations.labelstudio.data_source import LabelStudioDataSource
from flash.core.utilities.imports import _TEXT_AVAILABLE
from flash.text.data import (
    TextCSVDataSourceMixin,
    TextDataFrameDataSourceMixin,
    TextDataModule,
    TextDataSource,
    TextHuggingFaceDatasetDataSourceMixin,
    TextJSONDataSourceMixin,
    TextListDataSourceMixin,
    TextParquetDataSourceMixin,
    TextPreprocess,
)
from flash.text.tokenizers import BaseTokenizer
from flash.text.data import TokenizerState

if _TEXT_AVAILABLE:
    from transformers.modeling_outputs import SequenceClassifierOutput


class TextClassificationDataSource(TextDataSource):
    @staticmethod
    def _transform_label(label_to_class_mapping: Dict[str, int], target: str, ex: Dict[str, Union[int, str]]):
        ex[target] = label_to_class_mapping[ex[target]]
        return ex

    @staticmethod
    def _multilabel_target(targets: List[str], element: Dict[str, Any]) -> Dict[str, Any]:
        targets = [element.pop(target) for target in targets]
        element[DefaultDataKeys.TARGET] = targets
        return element

    def encode_target(self, hf_dataset, dataset, target) -> Sequence[Mapping[str, Any]]:
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
            hf_dataset = hf_dataset.rename_column(target, DefaultDataKeys.TARGET)

        return hf_dataset


class TextClassificationCSVDataSource(TextClassificationDataSource, TextCSVDataSourceMixin):
    pass


class TextClassificationJSONDataSource(TextClassificationDataSource, TextJSONDataSourceMixin):
    pass


class TextClassificationDataFrameDataSource(TextClassificationDataSource, TextDataFrameDataSourceMixin):
    pass


class TextClassificationParquetDataSource(TextClassificationDataSource, TextParquetDataSourceMixin):
    pass


class TextClassificationHuggingFaceDatasetDataSource(
    TextClassificationDataSource, TextHuggingFaceDatasetDataSourceMixin
):
    pass


class TextClassificationListDataSource(TextClassificationDataSource, TextListDataSourceMixin):
    def encode_target(self, hf_dataset, dataset, target) -> Sequence[Mapping[str, Any]]:
        if isinstance(target[0], List):
            # multi-target
            dataset.multi_label = True
            dataset.num_classes = len(target[0])
            self.set_state(LabelsState(target))
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

        return hf_dataset


class LabelStudioTextClassificationDataSource(LabelStudioDataSource):
    """The ``LabelStudioTextDataSource`` expects the input to
    :meth:`~flash.core.data.data_source.DataSource.load_data` to be a json export from label studio.
    Export data should point to text data
    """

    def __init__(self, tokenizer: BaseTokenizer):
        super().__init__()
        self.set_state(TokenizerState(tokenizer))

    def load_sample(self, sample: Mapping[str, Any] = None, dataset: Optional[Any] = None) -> Any:
        """Load 1 sample from dataset."""
        data = ""
        for key in sample.get("data"):
            data += sample.get("data").get(key)
        tokenized_data = self.get_state(TokenizerState).tokenizer(data)
        for key in tokenized_data:
            tokenized_data[key] = torch.tensor(tokenized_data[key])
        tokenized_data["labels"] = self._get_labels_from_sample(sample["label"])
        # separate text data type block
        result = tokenized_data
        return result


class TextClassificationPreprocess(TextPreprocess):
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

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources={
                DefaultDataSources.CSV: TextClassificationCSVDataSource,
                DefaultDataSources.JSON: TextClassificationJSONDataSource,
                DefaultDataSources.PARQUET: TextClassificationParquetDataSource,
                DefaultDataSources.HUGGINGFACE_DATASET: TextClassificationHuggingFaceDatasetDataSource,
                DefaultDataSources.DATAFRAME: TextClassificationDataFrameDataSource,
                DefaultDataSources.LISTS: TextClassificationListDataSource,
                DefaultDataSources.LABELSTUDIO: LabelStudioTextClassificationDataSource,
            },
            backbone=backbone,
            pretrained=pretrained,
            **backbone_kwargs,
        )


class TextClassificationPostprocess(Postprocess):
    def per_batch_transform(self, batch: Any) -> Any:
        if isinstance(batch, SequenceClassifierOutput):
            batch = batch.logits
        return super().per_batch_transform(batch)


class TextClassificationData(TextDataModule):
    """Data Module for text classification tasks."""

    preprocess_cls = TextClassificationPreprocess
    postprocess_cls = TextClassificationPostprocess
