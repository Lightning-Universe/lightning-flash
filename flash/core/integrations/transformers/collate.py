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
from typing import Any, Dict, Optional

import torch

from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _TRANSFORMERS_AVAILABLE

if _TRANSFORMERS_AVAILABLE:
    from transformers import AutoTokenizer


@dataclass(unsafe_hash=True)
class TransformersCollate:

    backbone: str
    tokenizer_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict, hash=False)

    def __post_init__(self):
        tokenizer_kwargs = self.tokenizer_kwargs or {}
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True, **tokenizer_kwargs)

    @staticmethod
    def to_tensor(sample: Dict[str, Any]) -> Dict[str, Any]:
        tensor_sample = {}
        for key in sample:
            if key is DataKeys.METADATA:
                tensor_sample[key] = sample[key]
            else:
                tensor_sample[key] = torch.tensor(sample[key])
        return tensor_sample

    def tokenize(self, sample):
        raise NotImplementedError

    def __call__(self, samples):
        return self.to_tensor(self.tokenize({key: [sample[key] for sample in samples] for key in samples[0].keys()}))
