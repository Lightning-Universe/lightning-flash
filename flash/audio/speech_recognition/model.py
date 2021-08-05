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
import warnings
from typing import Any, Callable, Dict, Mapping, Optional, Type, Union

import torch
import torch.nn as nn

from flash.audio.speech_recognition.backbone import SPEECH_RECOGNITION_BACKBONES
from flash.audio.speech_recognition.collate import DataCollatorCTCWithPadding
from flash.audio.speech_recognition.data import SpeechRecognitionBackboneState
from flash.core.data.process import Serializer
from flash.core.data.states import CollateFn
from flash.core.model import Task
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _AUDIO_AVAILABLE

if _AUDIO_AVAILABLE:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class SpeechRecognition(Task):

    backbones: FlashRegistry = SPEECH_RECOGNITION_BACKBONES

    required_extras = "audio"

    def __init__(
        self,
        backbone: str = "facebook/wav2vec2-base-960h",
        loss_fn: Optional[Callable] = None,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        learning_rate: float = 1e-5,
        serializer: Optional[Union[Serializer, Mapping[str, Serializer]]] = None,
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"
        # disable HF thousand warnings
        warnings.simplefilter("ignore")
        # set os environ variable for multiprocesses
        os.environ["PYTHONWARNINGS"] = "ignore"

        model = (
            self.backbones.get(backbone)() if backbone in self.backbones else Wav2Vec2ForCTC.from_pretrained(backbone)
        )
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            learning_rate=learning_rate,
            serializer=serializer,
        )

        self.save_hyperparameters()

        self.set_state(SpeechRecognitionBackboneState(backbone))
        self.set_state(CollateFn(DataCollatorCTCWithPadding(Wav2Vec2Processor.from_pretrained(backbone))))

    def forward(self, batch: Dict[str, torch.Tensor]):
        return self.model(batch["input_values"])

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self(batch)

    def step(self, batch: Any, batch_idx: int, metrics: nn.ModuleDict) -> Any:
        out = self.model(batch["input_values"], labels=batch["labels"])
        out["logs"] = {"loss": out.loss}
        return out
