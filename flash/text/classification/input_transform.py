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
from dataclasses import dataclass
from typing import Callable

from flash.core.data.io.input import DataKeys
from flash.core.integrations.transformers.input_transform import TransformersTextInputTransform
from flash.core.utilities.imports import _TRANSFORMERS_AVAILABLE

if _TRANSFORMERS_AVAILABLE:
    from transformers import DataCollatorWithPadding, default_data_collator


@dataclass(unsafe_hash=True)
class TextClassificationInputTransform(TransformersTextInputTransform):

    backbone: str = "prajjwal1/bert-medium"
    max_length: int = 128

    def tokenize_per_sample(self, sample):
        tokenized_sample = self._tokenize_call(
            text=sample[DataKeys.INPUT],
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            stride=self.stride,
        )
        tokenized_sample = tokenized_sample.data
        if DataKeys.TARGET in sample:
            tokenized_sample[DataKeys.TARGET] = sample[DataKeys.TARGET]
        return tokenized_sample

    def per_sample_transform(self) -> Callable:
        return self.tokenize_per_sample

    def collate(self) -> Callable:
        if self._is_pad_to_max_length():
            _collate_fn = default_data_collator
        else:
            _collate_fn = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=None)
        return _collate_fn
