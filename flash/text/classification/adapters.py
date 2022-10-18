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
from dataclasses import dataclass
from types import FunctionType
from typing import Any, Callable, Dict

import torch
from torch import Tensor

from flash.core.adapter import Adapter, AdapterTask
from flash.core.data.io.input import DataKeys
from flash.core.heads import CLASSIFIER_HEADS
from flash.core.model import Task
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _TRANSFORMERS_AVAILABLE
from flash.text.classification.collate import TextClassificationCollate

if _TRANSFORMERS_AVAILABLE:
    from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput, SequenceClassifierOutput


class HuggingFaceAdapter(Adapter):
    def __init__(self, backbone, num_classes: int, max_length: int = 128):
        super().__init__()

        os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"
        # disable HF thousand warnings
        warnings.simplefilter("ignore")
        # set os environ variable for multiprocesses
        os.environ["PYTHONWARNINGS"] = "ignore"

        self.model, tokenizer = backbone(num_classes)
        self.collate_fn = TextClassificationCollate(tokenizer, max_length=max_length)

    @classmethod
    def from_task(
        cls,
        task: AdapterTask,
        backbone: str,
        num_classes: int,
        **kwargs,
    ) -> Adapter:
        adapter = cls(backbone, num_classes, **kwargs)
        adapter.__dict__["_task"] = task
        return adapter

    @property
    def backbone(self):
        return self.model.base_model

    def forward(self, batch: Dict[str, Tensor]):
        result = self.model(input_ids=batch.get("input_ids", None), attention_mask=batch.get("attention_mask", None))
        if isinstance(result, (SequenceClassifierOutput, Seq2SeqSequenceClassifierOutput)):
            result = result.logits
        return result

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        target = batch.pop(DataKeys.TARGET)
        batch = (batch, target)
        return Task.training_step(self._task, batch, batch_idx)

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        target = batch.pop(DataKeys.TARGET)
        batch = (batch, target)
        return Task.validation_step(self._task, batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        target = batch.pop(DataKeys.TARGET)
        batch = (batch, target)
        return Task.test_step(self._task, batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self(batch)


@dataclass
class GenericCollate:

    tokenizer: Callable[[str], Any]

    @staticmethod
    def to_tensor(sample: Dict[str, Any]) -> Dict[str, Any]:
        tensor_sample = {}
        for key in sample:
            if key is DataKeys.METADATA:
                tensor_sample[key] = sample[key]
            else:
                tensor_sample[key] = torch.tensor(sample[key])
        return tensor_sample

    def tokenize(self, sample):
        sample[DataKeys.INPUT] = self.tokenizer(sample[DataKeys.INPUT])
        return sample

    def __call__(self, samples):
        return self.to_tensor(self.tokenize({key: [sample[key] for sample in samples] for key in samples[0].keys()}))


class GenericAdapter(Adapter):

    heads: FlashRegistry = CLASSIFIER_HEADS

    def __init__(self, backbone, num_classes: int, max_length: int = 128, head="linear"):
        super().__init__()

        self.backbone, tokenizer, num_features = backbone()

        self.collate_fn = GenericCollate(tokenizer)

        if isinstance(head, str):
            head = self.heads.get(head)(num_features=num_features, num_classes=num_classes)
        else:
            head = head(num_features, num_classes) if isinstance(head, FunctionType) else head

        self.head = head

    @classmethod
    def from_task(
        cls,
        task: AdapterTask,
        backbone: str,
        num_classes: int,
        **kwargs,
    ) -> Adapter:
        adapter = cls(backbone, num_classes, **kwargs)
        adapter.__dict__["_task"] = task
        return adapter

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DataKeys.INPUT], batch[DataKeys.TARGET])
        return Task.training_step(self._task, batch, batch_idx)

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DataKeys.INPUT], batch[DataKeys.TARGET])
        return Task.validation_step(self._task, batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DataKeys.INPUT], batch[DataKeys.TARGET])
        return Task.test_step(self._task, batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        batch[DataKeys.PREDS] = Task.predict_step(
            self._task, batch[DataKeys.INPUT], batch_idx, dataloader_idx=dataloader_idx
        )
        return batch

    def forward(self, x) -> Tensor:
        x = self.backbone(x)
        if x.dim() == 4:
            x = x.mean(-1).mean(-1)
        return self.head(x)
