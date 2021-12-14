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
from torch import Tensor

from flash.core.data.process import Deserializer
from flash.core.integrations.transformers.states import TransformersBackboneState
from flash.core.utilities.imports import requires


class TextDeserializer(Deserializer):
    @requires("text")
    def __init__(self, *args, max_length: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_length = max_length

    def serve_load_sample(self, text: str) -> Tensor:
        return self.get_state(TransformersBackboneState).tokenizer(
            text, max_length=self.max_length, truncation=True, padding="max_length"
        )

    @property
    def example_input(self) -> str:
        return "An example input"
