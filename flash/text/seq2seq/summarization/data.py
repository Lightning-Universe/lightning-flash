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
from typing import Any

from flash.core.data.process import Postprocess
from flash.core.utilities.imports import _TEXT_AVAILABLE
from flash.text.seq2seq.core.data import Seq2SeqBackboneState, Seq2SeqData, Seq2SeqPreprocess

if _TEXT_AVAILABLE:
    from transformers import AutoTokenizer


class SummarizationPostprocess(Postprocess):

    def __init__(self):
        super().__init__()

        if not _TEXT_AVAILABLE:
            raise ModuleNotFoundError("Please, pip install -e '.[text]'")

        self._backbone = None
        self._tokenizer = None

    @property
    def backbone(self):
        backbone_state = self.get_state(Seq2SeqBackboneState)
        if backbone_state is not None:
            return backbone_state.backbone

    @property
    def tokenizer(self):
        if self.backbone is not None and self.backbone != self._backbone:
            self._tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True)
            self._backbone = self.backbone
        return self._tokenizer

    def uncollate(self, generated_tokens: Any) -> Any:
        pred_str = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        pred_str = [str.strip(s) for s in pred_str]
        return pred_str


class SummarizationData(Seq2SeqData):

    preprocess_cls = Seq2SeqPreprocess
    postprocess_cls = SummarizationPostprocess
