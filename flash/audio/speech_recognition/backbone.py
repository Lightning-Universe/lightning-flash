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
from functools import partial

from flash.core.registry import ExternalRegistry, FlashRegistry
from flash.core.utilities.imports import _AUDIO_AVAILABLE
from flash.core.utilities.providers import _FAIRSEQ, _HUGGINGFACE

SPEECH_RECOGNITION_BACKBONES = FlashRegistry("backbones")

if _AUDIO_AVAILABLE:
    from transformers import AutoModelForCTC, Wav2Vec2ForCTC

    WAV2VEC_MODELS = ["facebook/wav2vec2-base-960h", "facebook/wav2vec2-large-960h-lv60"]

    for model_name in WAV2VEC_MODELS:
        SPEECH_RECOGNITION_BACKBONES(
            fn=partial(Wav2Vec2ForCTC.from_pretrained, model_name),
            name=model_name,
            providers=[_HUGGINGFACE, _FAIRSEQ],
        )

    HUGGINGFACE_BACKBONES = ExternalRegistry(AutoModelForCTC.from_pretrained, "backbones", providers=_HUGGINGFACE)

    SPEECH_RECOGNITION_BACKBONES += HUGGINGFACE_BACKBONES
