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
from dataclasses import dataclass
from typing import Any

import torch

from flash.core.data.io.output_transform import OutputTransform
from flash.core.data.properties import ProcessState
from flash.core.utilities.imports import _AUDIO_AVAILABLE, requires

if _AUDIO_AVAILABLE:
    from transformers import Wav2Vec2CTCTokenizer


@dataclass(unsafe_hash=True, frozen=True)
class SpeechRecognitionBackboneState(ProcessState):
    """The ``SpeechRecognitionBackboneState`` stores the backbone in use by the
    :class:`~flash.audio.speech_recognition.data.SpeechRecognitionOutputTransform`.
    """

    backbone: str


class SpeechRecognitionOutputTransform(OutputTransform):
    def __init__(self):
        super().__init__()

        self._backbone = None
        self._tokenizer = None

    @property
    def backbone(self):
        backbone_state = self.get_state(SpeechRecognitionBackboneState)
        if backbone_state is not None:
            return backbone_state.backbone

    @property
    def tokenizer(self):
        if self.backbone is not None and self.backbone != self._backbone:
            self._tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(self.backbone)
            self._backbone = self.backbone
        return self._tokenizer

    @requires("audio")
    def per_batch_transform(self, batch: Any) -> Any:
        # converts logits into greedy transcription
        pred_ids = torch.argmax(batch.logits, dim=-1)
        transcriptions = self.tokenizer.batch_decode(pred_ids)
        return transcriptions

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("_tokenizer", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.backbone is not None:
            self._tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(self.backbone)
