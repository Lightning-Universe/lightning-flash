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
from flash.core.registry import ExternalRegistry, FlashRegistry
from flash.core.utilities.imports import _TRANSFORMERS_AVAILABLE
from flash.core.utilities.providers import _HUGGINGFACE_TOKENIZERS
from flash.text.tokenizers.transformers import _trasformer_tokenizer

TEXT_CLASSIFIER_TOKENIZERS = FlashRegistry("tokenizers")

if _TRANSFORMERS_AVAILABLE:
    HUGGINGFACE_TEXT_CLASSIFIER_TOKENIZERS = ExternalRegistry(
        getter=_trasformer_tokenizer,
        name="trasformer",
        providers=_HUGGINGFACE_TOKENIZERS,
    )
    TEXT_CLASSIFIER_TOKENIZERS += HUGGINGFACE_TEXT_CLASSIFIER_TOKENIZERS
