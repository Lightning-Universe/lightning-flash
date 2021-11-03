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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from flash.core.data.data_pipeline import Postprocess
from flash.core.data.data_source import DefaultDataKeys, DefaultDataSources, LabelsState
from flash.core.integrations.labelstudio.data_source import LabelStudioTextClassificationDataSource
from flash.core.utilities.imports import _TEXT_AVAILABLE, requires
from flash.text.classification.tokenizers.base import BaseTokenizer
from flash.text.data import (
    TextCSVDataSourceMixin,
    TextDataFrameDataSourceMixin,
    TextDataModule,
    TextDataSource,
    TextDeserializer,
    TextHuggingFaceDatasetDataSourceMixin,
    TextJSONDataSourceMixin,
    TextListDataSourceMixin,
    TextParquetDataSourceMixin,
    TextPreprocessMixin,
)

if _TEXT_AVAILABLE:
    from transformers.modeling_outputs import SequenceClassifierOutput

    from flash.text.classification.tokenizers import TEXT_CLASSIFIER_TOKENIZERS


class TextClassificationDataSource(TextDataSource):
    @requires("text")
    def __init__(self, tokenizer: BaseTokenizer):
        super().__init__(tokenizer)

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

    def encode_input(self, hf_dataset, input) -> Sequence[Mapping[str, Any]]:
        # tokenize
        if not self.tokenizer._is_fit:
            self.tokenizer.fit(hf_dataset, input=input)
        hf_dataset = hf_dataset.map(partial(self._tokenize_fn, input=input), batched=True)
        hf_dataset = hf_dataset.remove_columns([input])  # just leave the numerical columns

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


class TextClassificationPreprocess(TextPreprocessMixin):
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
                DefaultDataSources.CSV: TextClassificationCSVDataSource(self.tokenizer),
                DefaultDataSources.JSON: TextClassificationJSONDataSource(self.tokenizer),
                DefaultDataSources.PARQUET: TextClassificationParquetDataSource(self.tokenizer),
                DefaultDataSources.HUGGINGFACE_DATASET: TextClassificationHuggingFaceDatasetDataSource(self.tokenizer),
                DefaultDataSources.DATAFRAME: TextClassificationDataFrameDataSource(self.tokenizer),
                DefaultDataSources.LISTS: TextClassificationListDataSource(self.tokenizer),
                DefaultDataSources.LABELSTUDIO: LabelStudioTextClassificationDataSource(self.tokenizer),
            },
            default_data_source=DefaultDataSources.LISTS,
            deserializer=TextDeserializer(self.tokenizer),
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
