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
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Type, Union

import torch
from pytorch_lightning.utilities import rank_zero_warn
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy, Metric

from flash.core.data.data_source import DefaultDataKeys
from flash.core.model import Task
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _IMAGE_AVAILABLE
from flash.core.utilities.isinstance import _isinstance
from flash.image.classification.data import ImageClassificationPreprocess

if _IMAGE_AVAILABLE:
    from flash.image.classification.backbones import IMAGE_CLASSIFIER_BACKBONES
else:
    IMAGE_CLASSIFIER_BACKBONES = FlashRegistry("backbones")


class ImageEmbedder(Task):
    """The ``ImageEmbedder`` is a :class:`~flash.Task` for obtaining feature vectors (embeddings) from images. For
    more details, see :ref:`image_embedder`.

    Args:
        embedding_dim: Dimension of the embedded vector. ``None`` uses the default from the backbone.
        backbone: A model to use to extract image features, defaults to ``"swav-imagenet"``.
        pretrained: Use a pretrained backbone, defaults to ``True``.
        loss_fn: Loss function for training and finetuning, defaults to :func:`torch.nn.functional.cross_entropy`
        optimizer: Optimizer to use for training and finetuning, defaults to :class:`torch.optim.SGD`.
        metrics: Metrics to compute for training and evaluation. Can either be an metric from the `torchmetrics`
            package, a custom metric inherenting from `torchmetrics.Metric`, a callable function or a list/dict
            containing a combination of the aforementioned. In all cases, each metric needs to have the signature
            `metric(preds,target)` and return a single scalar tensor. Defaults to :class:`torchmetrics.Accuracy`.
        learning_rate: Learning rate to use for training, defaults to ``1e-3``.
        pooling_fn: Function used to pool image to generate embeddings, defaults to :func:`torch.max`.
    """

    backbones: FlashRegistry = IMAGE_CLASSIFIER_BACKBONES

    required_extras: str = "image"

    def __init__(
        self,
        embedding_dim: Optional[int] = None,
        backbone: str = "resnet101",
        pretrained: bool = True,
        loss_fn: Callable = F.cross_entropy,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.SGD,
        metrics: Union[Metric, Callable, Mapping, Sequence, None] = (Accuracy()),
        learning_rate: float = 1e-3,
        pooling_fn: Callable = torch.max,
    ):
        super().__init__(
            model=None,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,
            preprocess=ImageClassificationPreprocess(),
        )

        self.save_hyperparameters()
        self.backbone_name = backbone
        self.embedding_dim = embedding_dim
        assert pooling_fn in [torch.mean, torch.max]
        self.pooling_fn = pooling_fn

        self.backbone, num_features = self.backbones.get(backbone)(pretrained=pretrained)

        if embedding_dim is None:
            self.head = nn.Identity()
        else:
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_features, embedding_dim),
            )
            rank_zero_warn("Adding linear layer on top of backbone. Remember to finetune first before using!")

    def apply_pool(self, x):
        x = self.pooling_fn(x, dim=-1)
        if _isinstance(x, Tuple[torch.Tensor, torch.Tensor]):
            x = x[0]
        x = self.pooling_fn(x, dim=-1)
        if _isinstance(x, Tuple[torch.Tensor, torch.Tensor]):
            x = x[0]
        return x

    def forward(self, x) -> torch.Tensor:
        x = self.backbone(x)

        # bolts ssl models return lists
        if isinstance(x, tuple):
            x = x[-1]

        if x.dim() == 4 and not self.embedding_dim:
            x = self.apply_pool(x)

        x = self.head(x)
        return x

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return super().test_step(batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        batch = batch[DefaultDataKeys.INPUT]
        return super().predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)
