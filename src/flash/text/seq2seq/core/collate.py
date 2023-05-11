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
from typing import Union

from flash.core.data.io.input import DataKeys
from flash.core.integrations.transformers.collate import TransformersCollate


@dataclass(unsafe_hash=True)
class TextSeq2SeqCollate(TransformersCollate):
    max_source_length: int = 128
    max_target_length: int = 128
    padding: Union[str, bool] = "max_length"

    def tokenize(self, sample):
        tokenized_sample = self.tokenizer(
            sample[DataKeys.INPUT],
            max_length=self.max_source_length,
            padding=self.padding,
            add_special_tokens=True,
            truncation=True,
        )
        tokenized_sample = tokenized_sample.data
        if DataKeys.TARGET in sample:
            with self.tokenizer.as_target_tokenizer():
                tokenized_sample[DataKeys.TARGET] = self.tokenizer(
                    sample[DataKeys.TARGET],
                    max_length=self.max_target_length,
                    padding=self.padding,
                    add_special_tokens=True,
                    truncation=True,
                )["input_ids"]
        return tokenized_sample
