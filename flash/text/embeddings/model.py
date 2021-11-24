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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torchmetrics
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_info
from sentence_transformers import SentenceTransformer
from torch import nn, Tensor
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import Metric

from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.process import Serializer
from flash.core.finetuning import FlashBaseFinetuning
from flash.core.model import Task
from flash.core.registry import FlashRegistry
from flash.text.embeddings.backbones import SENTENCE_TRANSFORMERS_BACKBONE


class SentenceEmbedder(Task):
    """The ``SentenceEmbedder`` is a :class:`~flash.Task` for generating sentence embeddings, training and
    validation. For more details, see `embeddings`.

    You can change the backbone to any question answering model from `UKPLab/sentence-transformers
    <https://github.com/UKPLab/sentence-transformers>`_ using the ``backbone``
    argument.

    .. note:: When changing the backbone, make sure you pass in the same backbone to the :class:`~flash.Task` and the
        :class:`~flash.core.data.data_module.DataModule` object! Since this is a Sentence Transformers task, make sure you
        use a Sentence Transformers model.

    Args:
        backbone: backbone model to use for the task.
        loss_fn: Loss function for training.
        optimizer: Optimizer to use for training, defaults to `torch.optim.Adam`.
        optimizer_kwargs: Additional kwargs to use when creating the optimizer (if not passed as an instance).
        scheduler: The scheduler or scheduler class to use.
        scheduler_kwargs: Additional kwargs to use when creating the scheduler (if not passed as an instance).
        metrics: Metrics to compute for training and evaluation. Defauls to calculating the ROUGE metric.
            Changing this argument currently has no effect.
        learning_rate: Learning rate to use for training, defaults to `3e-4`
        enable_ort: Enable Torch ONNX Runtime Optimization: https://onnxruntime.ai/docs/#onnx-runtime-for-training
    """

    required_extras: str = "text"

    backbones: FlashRegistry = SENTENCE_TRANSFORMERS_BACKBONE

    def __init__(
        self,
        backbone: str = "all-MiniLM-L6-v2",
        enable_ort: bool = False,
    ):

        os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"
        # disable HF thousand warnings
        warnings.simplefilter("ignore")
        # set os environ variable for multiprocesses
        os.environ["PYTHONWARNINGS"] = "ignore"
        super().__init__(
        )
        self.model = self.backbones.get(backbone)()

    def generate_embeddings(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
    ) -> Union[List[Tensor], np.ndarray, Tensor]:

        return self.model.encode(
            sentences=sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            output_value=output_value,
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
            device=device,
            normalize_embeddings=normalize_embeddings,
        )

    @property
    def backbone(self):
        return self.model.base_model

    def forward(self, x) -> torch.Tensor:
        """First call the backbone, then the model head."""
        x = self.backbone(x)
        return self.head(x)
