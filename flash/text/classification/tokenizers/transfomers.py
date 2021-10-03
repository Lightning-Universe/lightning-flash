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
from typing import Generator, List, Optional, Tuple, Union
from flash.core.data.data_source import DefaultDataKeys
from flash.text.classification.tokenizers.base import BaseTokenizer
import datasets
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from transformers import AutoConfig, AutoTokenizer


class TrasformerTokenizer(BaseTokenizer):

    # TODO: make this more flexible and allow users to pass tokenizers arguments!!

    def __init__(self, model_name: bool, pretrained: bool = True, **kwargs):
        self.model_name = model_name
        self.pretrained = pretrained

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # NOTE: self.tokenizer.model_max_length returns crazy value, pick this from config
        self.config = AutoConfig.from_pretrained(model_name)
        self.max_length = self.config.max_position_embeddings
        self.vocab_size = self.config.vocab_size
        self._is_fit = pretrained
        
        # Allow the user to specify this otherwise fallback to original
        if not pretrained:

            self.vocab_size = kwargs.get("vocab_size", self.config.vocab_size)
            self.batch_size = kwargs.get("batch_size", 1000)
            max_length = kwargs.get("max_length", self.max_length)
            if max_length > self.max_length:
                raise MisconfigurationException(
                    f"`max_length` must be less or equal to {self.max_length}, which is the maximum supported by {model_name}"
                )
            else:
                self.max_length = max_length

    def fit(self, batch_iterator: Generator[List[str], None, None]) -> None:
        if self._is_fit:
            return
        self.tokenizer = self.tokenizer.train_new_from_iterator(batch_iterator, vocab_size=self.vocab_size)
        self._is_fit = True

    def __call__(self, x: Union[str, List[str]], return_tensors: Optional[str] = None) -> Union[List[int], torch.Tensor]:
        if not self._is_fit:
            raise MisconfigurationException("If pretrained=False, tokenizer must be fit before using it")

        return self.tokenizer(
            x,
            return_token_type_ids=False,
            padding=True,  # pads to longest string in the batch, more efficient than "max_length"
            truncation=True,  # truncate to max_length supported by the model
            max_length=self.max_length,
            return_tensors=return_tensors,
        )

    def _batch_iterator(self, dataset: datasets.Dataset) -> Generator[List[str], None, None]:
        for i in range(0, len(dataset), self.batch_size):
            yield dataset[i : i + self.batch_size][DefaultDataKeys.INPUT]


def _trasformer_tokenizer(
    model_name: str = "prajjwal1/bert-tiny",
    pretrained: bool = True,
    **kwargs,
) -> Tuple["TrasformerTokenizer", int]:

    tokenizer = TrasformerTokenizer(model_name, pretrained, **kwargs)

    return tokenizer, tokenizer.vocab_size
