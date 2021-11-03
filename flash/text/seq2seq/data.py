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

from flash.core.data.data_source import DefaultDataKeys, DefaultDataSources
from flash.core.data.process import Postprocess
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
    from flash.text.classification.tokenizers import TEXT_CLASSIFIER_TOKENIZERS


class Seq2SeqDataSource(TextDataSource):
    @requires("text")
    def __init__(self, tokenizer: BaseTokenizer):
        super().__init__(tokenizer)

    def encode_input(self, hf_dataset, input) -> Sequence[Mapping[str, Any]]:
        # tokenize
        if not self.tokenizer._is_fit:
            self.tokenizer.fit(hf_dataset, input=input)
        hf_dataset = hf_dataset.map(partial(self._tokenize_fn, input=input), batched=True)
        hf_dataset = hf_dataset.remove_columns([input])  # just leave the numerical columns

        return hf_dataset

    def encode_target(self, hf_dataset, dataset, target) -> Sequence[Mapping[str, Any]]:
        if isinstance(target, List):
            target = DefaultDataKeys.TARGET

        hf_dataset = hf_dataset.map(
            lambda ex: {"target_input_ids": self._tokenize_fn(ex, target)["input_ids"]}, batched=True
        )
        hf_dataset = hf_dataset.remove_columns([target])  # just leave the numerical columns
        hf_dataset = hf_dataset.rename_column("target_input_ids", DefaultDataKeys.TARGET)

        return hf_dataset


class Seq2SeqCSVDataSource(Seq2SeqDataSource, TextCSVDataSourceMixin):
    pass


class Seq2SeqJSONDataSource(Seq2SeqDataSource, TextJSONDataSourceMixin):
    pass


class Seq2SeqDataFrameDataSource(Seq2SeqDataSource, TextDataFrameDataSourceMixin):
    pass


class Seq2SeqParquetDataSource(Seq2SeqDataSource, TextParquetDataSourceMixin):
    pass


class Seq2SeqHuggingFaceDatasetDataSource(Seq2SeqDataSource, TextHuggingFaceDatasetDataSourceMixin):
    pass


class Seq2SeqListDataSource(Seq2SeqDataSource, TextListDataSourceMixin):
    pass


class Seq2SeqPreprocess(TextPreprocessMixin):
    @requires("text")
    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        backbone: Union[str, Tuple[BaseTokenizer, int]] = "sshleifer/tiny-mbart",
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
                DefaultDataSources.CSV: Seq2SeqCSVDataSource(self.tokenizer),
                DefaultDataSources.JSON: Seq2SeqJSONDataSource(self.tokenizer),
                DefaultDataSources.PARQUET: Seq2SeqParquetDataSource(self.tokenizer),
                DefaultDataSources.HUGGINGFACE_DATASET: Seq2SeqHuggingFaceDatasetDataSource(self.tokenizer),
                DefaultDataSources.DATAFRAME: Seq2SeqDataFrameDataSource(self.tokenizer),
                DefaultDataSources.LISTS: Seq2SeqListDataSource(self.tokenizer),
            },
            default_data_source=DefaultDataSources.LISTS,
            deserializer=TextDeserializer(self.tokenizer),
        )


class Seq2SeqPostprocess(Postprocess):
    @requires("text")
    def __init__(self):
        super().__init__()

    def uncollate(self, generated_tokens: Any) -> Any:
        pred_str = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        pred_str = [str.strip(s) for s in pred_str]
        return pred_str


class Seq2SeqData(TextDataModule):
    """Data module for Seq2Seq tasks."""

    preprocess_cls = Seq2SeqPreprocess
    postprocess_cls = Seq2SeqPostprocess
