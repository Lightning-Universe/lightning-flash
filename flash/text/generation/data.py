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
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import Tensor

import flash
from flash.core.data.data_module import DataModule
from flash.core.data.data_source import DataSource, DefaultDataSources
from flash.core.data.process import Postprocess, Preprocess
from flash.core.data.properties import ProcessState
from flash.core.utilities.imports import _TEXT_AVAILABLE, requires_extras
from flash.text.classification.data import TextDeserializer

if _TEXT_AVAILABLE:
    import datasets
    from datasets import DatasetDict, load_dataset
    from transformers import GPT2Tokenizer, default_data_collator

class GPTDataSource(DataSource):
    @requires_extras("text")
    def __init__(
            self,
            backbone: str,
            max_source_length: int = 128,
            max_target_length: int = 128,
            padding: Union[str, bool] = "max_length",
    ):
        super().__init__()

        self.backbone = backbone
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.backbone, use_fast=True)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.padding = padding

    def _tokenize_fn(
        self,
        ex: Union[Dict[str, str], str],
        input: Optional[str] = None,
        target: Optional[str] = None,
    ) -> Callable:
        if isinstance(ex, dict):
            ex_input = ex[input]
            ex_target = ex[target] if target else None
        else:
            ex_input = ex
            ex_target = None
        return self.tokenizer()