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
import logging
import os
import warnings
from typing import Any, Dict, List, Optional

import torch
from pytorch_lightning import Callback
from torch import Tensor

from flash.core.model import Task
from flash.core.registry import FlashRegistry, print_provider_info
from flash.core.utilities.imports import _TEXT_AVAILABLE
from flash.core.utilities.providers import _SENTENCE_TRANSFORMERS
from flash.text.classification.collate import TextClassificationCollate
from flash.text.embedding.backbones import HUGGINGFACE_BACKBONES
from flash.text.ort_callback import ORTCallback

if _TEXT_AVAILABLE:
    from sentence_transformers.models import Pooling

    Pooling = print_provider_info("Pooling", _SENTENCE_TRANSFORMERS, Pooling)

logger = logging.getLogger(__name__)


class TextEmbedder(Task):
    """The ``TextEmbedder`` is a :class:`~flash.Task` for generating sentence embeddings, training and validation.
    For more details, see `embeddings`.

    You can change the backbone to any question answering model from `UKPLab/sentence-transformers
    <https://github.com/UKPLab/sentence-transformers>`_ using the ``backbone``
    argument.

    Args:
        backbone: backbone model to use for the task.
        enable_ort: Enable Torch ONNX Runtime Optimization: https://onnxruntime.ai/docs/#onnx-runtime-for-training
    """

    required_extras: str = "text"

    backbones: FlashRegistry = HUGGINGFACE_BACKBONES

    def __init__(
        self,
        backbone: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_length: int = 128,
        tokenizer_backbone: Optional[str] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        enable_ort: bool = False,
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"
        # disable HF thousand warnings
        warnings.simplefilter("ignore")
        # set os environ variable for multiprocesses
        os.environ["PYTHONWARNINGS"] = "ignore"
        super().__init__()

        if tokenizer_backbone is None:
            tokenizer_backbone = backbone
        self.max_length = max_length
        self.collate_fn = TextClassificationCollate(
            backbone=tokenizer_backbone, max_length=max_length, tokenizer_kwargs=tokenizer_kwargs
        )
        self.model = self.backbones.get(backbone)()
        self.pooling = Pooling(self.model.config.hidden_size)
        self.enable_ort = enable_ort

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        raise NotImplementedError("Training a `TextEmbedder` is not supported. Use a different text task instead.")

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        raise NotImplementedError("Validating a `TextEmbedder` is not supported. Use a different text task instead.")

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        raise NotImplementedError("Testing a `TextEmbedder` is not supported. Use a different text task instead.")

    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        """Adapted from sentence-transformers:

        https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Transformer.py#L45
        """
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(batch["input_ids"])

        trans_features = {"input_ids": batch["input_ids"], "attention_mask": attention_mask}
        if "token_type_ids" in batch:
            trans_features["token_type_ids"] = batch["token_type_ids"]

        output_states = self.model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        batch.update({"token_embeddings": output_tokens, "attention_mask": attention_mask})

        return self.pooling(batch)["sentence_embedding"]

    def configure_callbacks(self) -> List[Callback]:
        callbacks = super().configure_callbacks() or []
        if self.enable_ort:
            callbacks.append(ORTCallback())
        return callbacks
