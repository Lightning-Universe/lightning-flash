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
from typing import Any, Callable, Mapping, Optional, Sequence, Type, Union

import torch
from pytorch_lightning.utilities.distributed import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy

from flash.core import Task
from flash.core.data import TaskDataPipeline
from flash.core.data.utils import _contains_any_tensor
from flash.vision.backbones import backbone_and_num_features
from flash.vision.classification.data import _DEFAULT_VALID_TRANSFORMS, _pil_loader


class ImageEmbedderDataPipeline(TaskDataPipeline):
    """
    >>> from flash.vision.embedding import ImageEmbedderDataPipeline
    >>> iedata = ImageEmbedderDataPipeline()
    >>> iedata.before_collate(torch.tensor([1]))
    tensor([1])
    >>> import os, numpy, PIL
    >>> img = PIL.Image.fromarray(numpy.random.randint(0, 255, (150, 200, 3)), 'RGB')
    >>> img.save('sample-image.png')
    >>> iedata.before_collate('sample-image.png')  # doctest: +ELLIPSIS
    [tensor([[[...]]])]
    >>> os.remove('sample-image.png')
    """

    def __init__(
        self,
        valid_transform: Optional[Callable] = _DEFAULT_VALID_TRANSFORMS,
        loader: Callable = _pil_loader,
    ):
        self._valid_transform = valid_transform
        self._loader = loader

    def before_collate(self, samples: Any) -> Any:
        if _contains_any_tensor(samples):
            return samples

        if isinstance(samples, str):
            samples = [samples]

        if isinstance(samples, (list, tuple)) and all(isinstance(p, str) for p in samples):
            outputs = []
            for sample in samples:
                output = self._loader(sample)
                outputs.append(self._valid_transform(output))
            return outputs
        raise MisconfigurationException("The samples should either be a tensor, a list of paths or a path.")


class ImageEmbedder(Task):
    """Task that classifies images.

    Args:
        embedding_dim: Dimension of the embedded vector. ``None`` uses the default from the backbone.
        backbone: A model to use to extract image features, defaults to ``"swav-imagenet"``.
        pretrained: Use a pretrained backbone, defaults to ``True``.
        loss_fn: Loss function for training and finetuning, defaults to :func:`torch.nn.functional.cross_entropy`
        optimizer: Optimizer to use for training and finetuning, defaults to :class:`torch.optim.SGD`.
        metrics: Metrics to compute for training and evaluation.
        learning_rate: Learning rate to use for training, defaults to ``1e-3``.
        pooling_fn: Function used to pool image to generate embeddings, defaults to :func:`torch.max`.

    Example:
        >>> import torch
        >>> from flash.vision.embedding import ImageEmbedder
        >>> embedder = ImageEmbedder(backbone='resnet18')
        >>> image = torch.rand(32, 3, 32, 32)
        >>> embeddings = embedder(image)

    """

    def __init__(
        self,
        embedding_dim: Optional[int] = None,
        backbone: str = "swav-imagenet",
        pretrained: bool = True,
        loss_fn: Callable = F.cross_entropy,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.SGD,
        metrics: Union[Callable, Mapping, Sequence, None] = (Accuracy()),
        learning_rate: float = 1e-3,
        pooling_fn: Callable = torch.max
    ):
        super().__init__(
            model=None,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,
        )

        self.save_hyperparameters()
        self.backbone_name = backbone
        self.embedding_dim = embedding_dim
        assert pooling_fn in [torch.mean, torch.max]
        self.pooling_fn = pooling_fn

        self.backbone, num_features = backbone_and_num_features(backbone, pretrained=pretrained)

        if embedding_dim is None:
            self.head = nn.Identity()
        else:
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_features, embedding_dim),
            )
            rank_zero_warn('embedding_dim is not None. Remember to finetune first!')

    def apply_pool(self, x):
        if self.pooling_fn == torch.max:
            # torch.max also returns argmax
            x = self.pooling_fn(x, dim=-1)[0]
            x = self.pooling_fn(x, dim=-1)[0]
        else:
            x = self.pooling_fn(x, dim=-1)
            x = self.pooling_fn(x, dim=-1)
        return x

    def forward(self, x) -> Any:
        x = self.backbone(x)

        # bolts ssl models return lists
        if isinstance(x, tuple):
            x = x[-1]

        if x.dim() == 4 and self.embedding_dim is not None:
            x = self.apply_pool(x)

        x = self.head(x)
        return x

    @staticmethod
    def default_pipeline() -> ImageEmbedderDataPipeline:
        return ImageEmbedderDataPipeline()
