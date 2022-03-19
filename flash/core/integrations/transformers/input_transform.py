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
from typing import Any, Dict, List, Optional, Union

import torch

from flash.core.data.io.input import DataKeys
from flash.core.data.io.input_transform import InputTransform
from flash.core.utilities.imports import _TRANSFORMERS_AVAILABLE

if _TRANSFORMERS_AVAILABLE:
    from transformers import AutoTokenizer
    from transformers.file_utils import PaddingStrategy
    from transformers.tokenization_utils_base import PreTokenizedInput, TextInput, TruncationStrategy
else:
    AutoTokenizer = object
    PaddingStrategy = object
    PreTokenizedInput = object
    TextInput = object
    TruncationStrategy = object


@dataclass(unsafe_hash=True)
class TransformersTextInputTransform(InputTransform):

    backbone: str
    max_length: Optional[int] = None
    padding: Union[bool, str, PaddingStrategy] = "max_length"
    truncation: Union[bool, str, TruncationStrategy] = True
    stride: int = 0
    add_special_tokens: bool = True
    use_fast: bool = True

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=self.use_fast)
        super().__post_init__()

    def _tokenize_call(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        **tokenixer_call_kwargs
    ):
        return self.tokenizer(text=text, text_pair=text_pair, **tokenixer_call_kwargs)

    def _is_pad_to_max_length(self) -> bool:
        return self.padding == "max_length" or self.padding == PaddingStrategy.MAX_LENGTH

    @staticmethod
    def to_tensor(sample: Dict[str, Any]) -> Dict[str, Any]:
        tensor_sample = {}
        for key in sample:
            if key is DataKeys.METADATA:
                tensor_sample[key] = sample[key]
            else:
                tensor_sample[key] = torch.tensor(sample[key])
        return tensor_sample
