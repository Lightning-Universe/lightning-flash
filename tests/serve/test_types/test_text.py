from dataclasses import dataclass

import pytest
import torch

from flash.core.utilities.imports import _TRANSFORMERS_AVAILABLE


@dataclass
class CustomTokenizer:
    name: str

    def encode(self, text, return_tensors="pt"):
        return f"encoding from {self.name}"

    def decode(self, tensor):
        return f"decoding from {self.name}"


@pytest.mark.skipif(not _TRANSFORMERS_AVAILABLE, reason="the library transformers is not installed.")
def test_custom_tokenizer():
    from flash.core.serve.types import Text

    tokenizer = CustomTokenizer("test")
    text = Text(tokenizer=tokenizer)
    assert "encoding from test" == text.deserialize("random string")
    assert "decoding from test" == text.serialize(torch.tensor([[1, 2]]))


@pytest.mark.skipif(not _TRANSFORMERS_AVAILABLE, reason="the library transformers is not installed.")
def test_tokenizer_string():
    from flash.core.serve.types import Text

    text = Text(tokenizer="google/pegasus-xsum")
    assert torch.allclose(torch.tensor([[181, 4211, 1]], dtype=torch.long), text.deserialize("some string"))
    assert "</s><mask_1>" == text.serialize(torch.tensor([[1, 2]]))
