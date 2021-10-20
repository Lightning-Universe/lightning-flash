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


class CrossEncoder(Task):
    """The ``CrossEncoder`` is a :class:`~flash.Task` for generating sentence embeddings, training and validation.
    For more details, see `cross_encoders`.

    You can change the backbone to any CrossEncoder model from `UKPLab/sentence-transformers
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
        backbone: str = "distilroberta-base",
        loss_fn: Optional[Union[Callable, Mapping, Sequence]] = None,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Union[Type[_LRScheduler], str, _LRScheduler]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        metrics: Union[Metric, Callable, Mapping, Sequence, None] = None,
        learning_rate: float = 5e-5,
        enable_ort: bool = False,
    ):

        os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"
        # disable HF thousand warnings
        warnings.simplefilter("ignore")
        # set os environ variable for multiprocesses
        os.environ["PYTHONWARNINGS"] = "ignore"
        super().__init__(
            loss_fn=loss_fn,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            metrics=metrics,
            learning_rate=learning_rate,
        )
        self.model = self.backbones.get(backbone)()

    @property
    def backbone(self):
        return self.model.base_model

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        """For the training step, we just extract the :attr:`~flash.core.data.data_source.DefaultDataKeys.INPUT` and
        :attr:`~flash.core.data.data_source.DefaultDataKeys.TARGET` keys from the input and forward them to the
        :meth:`~flash.core.model.Task.training_step`."""
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        """For the validation step, we just extract the :attr:`~flash.core.data.data_source.DefaultDataKeys.INPUT` and
        :attr:`~flash.core.data.data_source.DefaultDataKeys.TARGET` keys from the input and forward them to the
        :meth:`~flash.core.model.Task.validation_step`."""
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        """For the test step, we just extract the :attr:`~flash.core.data.data_source.DefaultDataKeys.INPUT` and
        :attr:`~flash.core.data.data_source.DefaultDataKeys.TARGET` keys from the input and forward them to the
        :meth:`~flash.core.model.Task.test_step`."""
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return super().test_step(batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """For the predict step, we just extract the :attr:`~flash.core.data.data_source.DefaultDataKeys.INPUT` key
        from the input and forward it to the :meth:`~flash.core.model.Task.predict_step`."""
        return self.model.predict(batch)

    def forward(self, x) -> torch.Tensor:
        """First call the backbone, then the model head."""
        x = self.backbone(x)
        return self.head(x)
