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
from typing import Any, Dict, Optional, Type, Union

import torch.nn as nn
from torch import Tensor

from flash.audio.speech_recognition.backbone import SPEECH_RECOGNITION_BACKBONES
from flash.audio.speech_recognition.collate import DataCollatorCTCWithPadding
from flash.audio.speech_recognition.input import SpeechRecognitionDeserializer
from flash.audio.speech_recognition.output_transform import SpeechRecognitionOutputTransform
from flash.core.data.io.input import ServeInput
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.io.output import Output
from flash.core.model import Task
from flash.core.registry import FlashRegistry
from flash.core.serve import Composition
from flash.core.utilities.imports import _AUDIO_AVAILABLE, requires
from flash.core.utilities.types import INPUT_TRANSFORM_TYPE, LR_SCHEDULER_TYPE, OPTIMIZER_TYPE

if _AUDIO_AVAILABLE:
    from transformers import AutoProcessor


class SpeechRecognition(Task):
    """The ``SpeechRecognition`` task is a :class:`~flash.Task` for converting speech to text. For more details, see
    :ref:`speech_recognition`.

    Args:
        backbone: Any speech recognition model from `HuggingFace/transformers
            <https://huggingface.co/models?pipeline_tag=automatic-speech-recognition>`_.
        learning_rate: Learning rate to use for training, defaults to ``1e-5``.
        optimizer: Optimizer to use for training.
        lr_scheduler: The LR scheduler to use during training.
    """

    backbones: FlashRegistry = SPEECH_RECOGNITION_BACKBONES

    required_extras = "audio"

    def __init__(
        self,
        backbone: str = "facebook/wav2vec2-base-960h",
        processor_backbone: str = None,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        learning_rate: Optional[float] = None,
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
            lr_scheduler=lr_scheduler,
            learning_rate=learning_rate,
            output_transform=SpeechRecognitionOutputTransform(backbone)
            if processor_backbone is None
            else SpeechRecognitionOutputTransform(processor_backbone),
        )

        self.save_hyperparameters()

        self.collate_fn = DataCollatorCTCWithPadding(
            AutoProcessor.from_pretrained(backbone)
            if processor_backbone is None
            else AutoProcessor.from_pretrained(processor_backbone)
        )

    def modules_to_freeze(self) -> Optional[nn.Module]:
        return self.model.base_model

    def forward(self, batch: Dict[str, Tensor]):
        return self.model(batch["input_values"]).logits

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self(batch)

    def step(self, batch: Any, batch_idx: int, metrics: nn.ModuleDict) -> Any:
        out = self.model(batch["input_values"], labels=batch["labels"])
        out["logs"] = {"loss": out.loss}
        return out

    @requires("serve")
    def serve(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        sanity_check: bool = True,
        input_cls: Optional[Type[ServeInput]] = SpeechRecognitionDeserializer,
        transform: INPUT_TRANSFORM_TYPE = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        output: Optional[Union[str, Output]] = None,
    ) -> Composition:
        return super().serve(host, port, sanity_check, input_cls, transform, transform_kwargs, output)
