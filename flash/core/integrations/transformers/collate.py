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
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Optional

import torch
from torch.utils.data._utils.collate import default_collate

from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _TRANSFORMERS_AVAILABLE

if _TRANSFORMERS_AVAILABLE:
    from transformers import AutoTokenizer


@dataclass(unsafe_hash=True, frozen=True)
class TransformersCollate:

    backbone: str
    max_length: int = (128,)
    tokenizer_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict, hash=False)

    @staticmethod
    def to_tensor(sample: Dict[str, Any]) -> Dict[str, Any]:
        for key in sample:
            if key is DataKeys.METADATA:
                continue
            sample[key] = torch.as_tensor(sample[key])
        return sample

    @property
    @lru_cache(maxsize=None)
    def tokenizer(self):
        tokenizer_kwargs = {}
        if self.tokenizer_kwargs is not None:
            tokenizer_kwargs = self.tokenizer_kwargs
        return AutoTokenizer.from_pretrained(self.backbone, use_fast=True, **tokenizer_kwargs)

    def __call__(self, samples):
        tokenized_samples = []
        for sample in samples:
            tokenized_sample = self.tokenizer(
                sample[DataKeys.INPUT], max_length=self.max_length, truncation=True, padding="max_length"
            )
            tokenized_sample = tokenized_sample.data
            if DataKeys.TARGET in sample:
                tokenized_sample[DataKeys.TARGET] = sample[DataKeys.TARGET]
            tokenized_samples.append(self.to_tensor(tokenized_sample))
        return default_collate(tokenized_samples)
