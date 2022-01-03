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

from flash.core.data.io.output_transform import OutputTransform
from flash.core.integrations.transformers.states import TransformersBackboneState
from flash.core.utilities.imports import requires


class Seq2SeqOutputTransform(OutputTransform):
    def __init__(self):
        super().__init__()

        self._backbone = None
        self._tokenizer = None

    @requires("text")
    def uncollate(self, generated_tokens: Any) -> Any:
        tokenizer = self.get_state(TransformersBackboneState).tokenizer
        pred_str = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        pred_str = [str.strip(s) for s in pred_str]
        return pred_str
