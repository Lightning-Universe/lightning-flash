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

from transformers import AutoTokenizer

from flash.data.process import Postprocess
from flash.text.seq2seq.core.data import Seq2SeqData, Seq2SeqPreprocess


class SummarizationPostprocess(Postprocess):

    def __init__(
        self,
        backbone: str = "sshleifer/tiny-mbart",
    ):
        super().__init__()

        # TODO: Should share the backbone or tokenizer over state
        self.tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)

    def uncollate(self, generated_tokens: Any) -> Any:
        pred_str = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        pred_str = [str.strip(s) for s in pred_str]
        return pred_str


class SummarizationData(Seq2SeqData):

    preprocess_cls = Seq2SeqPreprocess
    postprocess_cls = SummarizationPostprocess
