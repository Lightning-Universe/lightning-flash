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

from flash.core.data.io.input import DataKeys
from flash.core.integrations.transformers.collate import TransformersCollate


@dataclass(unsafe_hash=True)
class TextClassificationCollate(TransformersCollate):

    max_length: int = 128

    def tokenize(self, sample):
        tokenized_sample = self.tokenizer(
            sample[DataKeys.INPUT], max_length=self.max_length, truncation=True, padding="max_length"
        )
        tokenized_sample = tokenized_sample.data
        if DataKeys.TARGET in sample:
            tokenized_sample[DataKeys.TARGET] = sample[DataKeys.TARGET]
        return tokenized_sample
