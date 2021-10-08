from typing import Generator, List, Optional, Tuple, Union

import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from transformers import AutoConfig, AutoTokenizer

from flash.text.classification.tokenizers.base import BaseTokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Sequence, Digits, Whitespace
from tokenizers.normalizers import BertNormalizer
from transfomers import PreTrainedTokenizerFast


class WhiteSpaceTokenizer(BaseTokenizer, PreTrainedTokenizerFast):

    def __init__(self, backbone: str, **backbone_kwargs):
        super().__init__(backbone, False)

        self.unk_token = "[UNK]"
        self.tokenizer = Tokenizer(WordLevel(unk_token=self.unk_token))
        self.tokenizer.pre_tokenizer = Sequence([Digits(), Whitespace()])
        self.tokenizer.normalizer = BertNormalizer()
        self.trainer = WordLevelTrainer(vocab_size=3, special_tokens=[self.unk_token])
 
        self.vocab_size = backbone_kwargs.get("vocab_size", 50000)
        self.max_length = backbone_kwargs.get("max_length", 1000)
        self.batch_size = backbone_kwargs.get("batch_size", 1000)

    def fit(self, batch_iterator: Generator[List[str], None, None]) -> None:
        if self._is_fit:
            return
        self.tokenizer.train_from_iterator(batch_iterator, trainer=self.trainer)  # in-place
        self._is_fit = True

    def __call__(
        self, x: Union[str, List[str]], return_tensors: Optional[str] = None
    ) -> Union[List[int], torch.Tensor]:
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


def _trasformer_tokenizer(
    backbone: str = "prajjwal1/bert-tiny",
    pretrained: bool = True,
    **backbone_kwargs,
) -> Tuple["TransformerTokenizer", int]:

    tokenizer = TransformerTokenizer(backbone, pretrained, **backbone_kwargs)

    return tokenizer, tokenizer.vocab_size