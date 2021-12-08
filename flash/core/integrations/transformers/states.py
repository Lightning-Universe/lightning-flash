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

from flash.core.data.properties import ProcessState
from flash.core.utilities.imports import _TRANSFORMERS_AVAILABLE

if _TRANSFORMERS_AVAILABLE:
    from transformers import AutoTokenizer


@dataclass(unsafe_hash=True, frozen=True)
class TransformersBackboneState(ProcessState):
    """The ``TransformersBackboneState`` records the ``backbone`` in use by tasks which rely on Hugging Face
    transformers."""

    backbone: str
    tokenizer_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict, hash=False)

    @property
    @lru_cache(maxsize=None)
    def tokenizer(self):
        tokenizer_kwargs = {}
        if self.tokenizer_kwargs is not None:
            tokenizer_kwargs = self.tokenizer_kwargs
        return AutoTokenizer.from_pretrained(self.backbone, use_fast=True, **tokenizer_kwargs)
