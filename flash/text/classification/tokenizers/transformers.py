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
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from transformers import AutoConfig, AutoTokenizer

from flash.text.classification.tokenizers.base import BaseTokenizer


class TransformerTokenizer(BaseTokenizer):
    def __init__(self, backbone: str, pretrained: bool = True, **backbone_kwargs):
        super().__init__(backbone, pretrained)

        self.tokenizer = AutoTokenizer.from_pretrained(backbone)

        # NOTE: self.tokenizer.model_max_length returns crazy value, pick this from config
        self.config = AutoConfig.from_pretrained(backbone)
        self.max_length = self.config.max_position_embeddings
        self.vocab_size = self.config.vocab_size
        self.return_token_type_ids = backbone_kwargs.get("return_token_type_ids", False)
        self.padding = backbone_kwargs.get("padding", True)
        self.truncation = backbone_kwargs.get("truncation", True)

        # max_length
        max_length = backbone_kwargs.get("max_length", self.max_length)
        if max_length > self.max_length:
            raise MisconfigurationException(
                f"`max_length` must be less or equal to {self.max_length}, which is the maximum supported by {backbone}"
            )
        else:
            self.max_length = max_length

        # allow the user to specify this otherwise fallback to original
        if not pretrained:
            self.vocab_size = backbone_kwargs.get("vocab_size", self.config.vocab_size)
            self.batch_size = backbone_kwargs.get("batch_size", 1000)

    def fit(self, hf_dataset: Sequence[Mapping[str, Any]], input: str) -> None:
        if self._is_fit:
            print("Tokenizer is already trained. Doint nothing.")
            return
        batch_iterator = self._batch_iterator(hf_dataset, input=input)
        self.tokenizer = self.tokenizer.train_new_from_iterator(batch_iterator, vocab_size=self.vocab_size)
        self._is_fit = True

    def __call__(self, x: Union[str, List[str]]) -> Union[List[int], torch.Tensor]:
        if not self._is_fit:
            raise MisconfigurationException("If pretrained=False, tokenizer must be fit before using it")

        return self.tokenizer(
            x,
            return_token_type_ids=self.return_token_type_ids,
            padding=self.padding,  # pads to longest string in the batch, more efficient than "max_length"
            truncation=self.truncation,  # truncate to max_length supported by the model
            max_length=self.max_length,
            return_tensors=None,  # transfomed to tensor in `load_data`
        )


def _trasformer_tokenizer(
    backbone: str = "prajjwal1/bert-tiny",
    pretrained: bool = True,
    **backbone_kwargs,
) -> Tuple["TransformerTokenizer", int]:

    tokenizer = TransformerTokenizer(backbone, pretrained, **backbone_kwargs)

    return tokenizer, tokenizer.vocab_size
