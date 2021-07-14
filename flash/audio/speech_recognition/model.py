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

from flash import Task
from flash.core.data.process import Serializer
from flash.core.utilities.imports import _TEXT_AVAILABLE

if _TEXT_AVAILABLE:
    from transformers import Wav2Vec2ForCTC


class SpeechRecognition(Task):

    def __init__(
        self,
        backbone: str = "facebook/wav2vec2-base-960h",
        loss_fn: Optional[Callable] = None,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        learning_rate: float = 1e-2,
        serializer: Optional[Union[Serializer, Mapping[str, Serializer]]] = None,
    ):
        self.save_hyperparameters()

        os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"
        # disable HF thousand warnings
        warnings.simplefilter("ignore")
        # set os environ variable for multiprocesses
        os.environ["PYTHONWARNINGS"] = "ignore"

        super().__init__(
            model=None,
            loss_fn=loss_fn,
            optimizer=optimizer,
            learning_rate=learning_rate,
            serializer=serializer,
        )
        self.model = Wav2Vec2ForCTC.from_pretrained(backbone)

        self.save_hyperparameters()

    def forward(self, batch: Dict[str, torch.Tensor]):
        return self.model(input_ids=batch["input_ids"])

    def step(self, batch, batch_idx, metrics) -> dict:
        out = self(batch)
        prediction = torch.argmax(out.logits, dim=-1)
        return prediction
