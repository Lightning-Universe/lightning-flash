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

import torch

from flash.core.data.io.output_transform import OutputTransform
from flash.core.utilities.imports import _AUDIO_AVAILABLE, requires

if _AUDIO_AVAILABLE:
    from transformers import Wav2Vec2CTCTokenizer


class SpeechRecognitionOutputTransform(OutputTransform):
    def __init__(self, backbone: str):
        super().__init__()

        self.backbone = backbone
        self._tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(self.backbone)

    @requires("audio")
    def per_batch_transform(self, batch: Any) -> Any:
        # converts logits into greedy transcription
        pred_ids = torch.argmax(batch, dim=-1)
        transcriptions = self._tokenizer.batch_decode(pred_ids)
        return transcriptions

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("_tokenizer", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.backbone is not None:
            self._tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(self.backbone)
