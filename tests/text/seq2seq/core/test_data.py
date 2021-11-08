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
import os

import pytest

from flash.core.utilities.imports import _TEXT_AVAILABLE
from flash.text.seq2seq.core.data import (
    Seq2SeqBackboneState,
    Seq2SeqCSVDataSource,
    Seq2SeqDataSource,
    Seq2SeqFileDataSource,
    Seq2SeqJSONDataSource,
    Seq2SeqOutputTransform,
    Seq2SeqSentencesDataSource,
)
from tests.helpers.utils import _TEXT_TESTING

if _TEXT_AVAILABLE:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (Seq2SeqDataSource, {"backbone": "sshleifer/tiny-mbart"}),
        (Seq2SeqFileDataSource, {"backbone": "sshleifer/tiny-mbart", "filetype": "csv"}),
        (Seq2SeqCSVDataSource, {"backbone": "sshleifer/tiny-mbart"}),
        (Seq2SeqJSONDataSource, {"backbone": "sshleifer/tiny-mbart"}),
        (Seq2SeqSentencesDataSource, {"backbone": "sshleifer/tiny-mbart"}),
        (Seq2SeqOutputTransform, {}),
    ],
)
def test_tokenizer_state(cls, kwargs):
    """Tests that the tokenizer is not in __getstate__"""
    process_state = Seq2SeqBackboneState(backbone="sshleifer/tiny-mbart")
    instance = cls(**kwargs)
    instance.set_state(process_state)
    getattr(instance, "tokenizer", None)
    state = instance.__getstate__()
    tokenizers = []
    for name, attribute in instance.__dict__.items():
        if isinstance(attribute, PreTrainedTokenizerBase):
            assert name not in state
            setattr(instance, name, None)
            tokenizers.append(name)
    instance.__setstate__(state)
    for name in tokenizers:
        assert getattr(instance, name, None) is not None
