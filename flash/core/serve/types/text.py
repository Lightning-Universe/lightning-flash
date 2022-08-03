import warnings
from dataclasses import dataclass
from typing import Any, Union

from torch import Tensor

from flash.core.serve.types.base import BaseType


@dataclass(unsafe_hash=True)
class Text(BaseType):
    """Type for converting string to tensor and back.

    Parameters
    ----------
    tokenizer: Union[str, Any]
        Tokenizer to convert input text to indices. If the tokenizer is string,
        it will be loaded using Huggingface transformers' AutoTokenizer and hence
        must be available here https://huggingface.co/models. If it's an object,
        it needs to have `encode` and `decode` method for deserialization and
        serialization respectively.

    TODO: Allow other arguments such as language, max_len etc. Add guidelines
     to write custom tokenizer
    """

    tokenizer: Union[str, Any]

    def __post_init__(self):
        if isinstance(self.tokenizer, str):
            try:
                from transformers import AutoTokenizer
            except (ImportError, ModuleNotFoundError) as e:  # pragma: no cover
                msg = "install the 'transformers' package to make use of this feature"
                warnings.warn(msg, UserWarning)
                raise e
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)

    def deserialize(self, text: str) -> Tensor:
        return self.tokenizer.encode(text, return_tensors="pt")

    def serialize(self, tensor: Tensor) -> str:
        return self.tokenizer.decode(tensor.squeeze())
