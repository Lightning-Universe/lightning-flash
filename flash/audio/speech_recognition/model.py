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
from typing import Any, Dict, Mapping, Optional, Type, Union

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

from flash.audio.speech_recognition.backbone import SPEECH_RECOGNITION_BACKBONES
from flash.audio.speech_recognition.collate import DataCollatorCTCWithPadding
from flash.audio.speech_recognition.data import SpeechRecognitionBackboneState
from flash.core.data.process import Serializer
from flash.core.data.states import CollateFn
from flash.core.model import Task
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _AUDIO_AVAILABLE

if _AUDIO_AVAILABLE:
    from transformers import Wav2Vec2Processor


class SpeechRecognition(Task):
    """The ``SpeechRecognition`` task is a :class:`~flash.Task` for converting speech to text. For more details, see
    :ref:`speech_recognition`.

    Args:
        backbone: Any speech recognition model from `HuggingFace/transformers
            <https://huggingface.co/models?pipeline_tag=automatic-speech-recognition>`_.
        optimizer: Optimizer to use for training.
        optimizer_kwargs: Additional kwargs to use when creating the optimizer (if not passed as an instance).
        scheduler: The scheduler or scheduler class to use.
        scheduler_kwargs: Additional kwargs to use when creating the scheduler (if not passed as an instance).
        learning_rate: Learning rate to use for training, defaults to ``1e-3``.
        serializer: The :class:`~flash.core.data.process.Serializer` to use when serializing prediction outputs.
    """

    backbones: FlashRegistry = SPEECH_RECOGNITION_BACKBONES

    required_extras = "audio"

    def __init__(
        self,
        backbone: str = "facebook/wav2vec2-base-960h",
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Union[Type[_LRScheduler], str, _LRScheduler]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        learning_rate: float = 1e-5,
        serializer: Optional[Union[Serializer, Mapping[str, Serializer]]] = None,
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"
        # disable HF thousand warnings
        warnings.simplefilter("ignore")
        # set os environ variable for multiprocesses
        os.environ["PYTHONWARNINGS"] = "ignore"

        model = self.backbones.get(backbone)()
        super().__init__(
            model=model,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
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
