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
from typing import Any, Callable, Mapping, Sequence, Type, Union

import torch
import torchvision
from pytorch_lightning.metrics import Accuracy
from torch import nn
from torch.nn import functional as F
from pytorch_lightning.utilities.distributed import rank_zero_warn

from flash.core import Task
from flash.vision.classification.data import ImageClassificationData, ImageClassificationDataPipeline
from flash.model_map import models

_resnet_backbone = lambda model: nn.Sequential(*list(model.children())[:-2])  # noqa: E731
_resnet_feats = lambda model: model.fc.in_features  # noqa: E731

_backbones = {
    "resnet18": (torchvision.models.resnet18, _resnet_backbone, _resnet_feats),
    "resnet34": (torchvision.models.resnet34, _resnet_backbone, _resnet_feats),
    "resnet50": (torchvision.models.resnet50, _resnet_backbone, _resnet_feats),
    "resnet101": (torchvision.models.resnet101, _resnet_backbone, _resnet_feats),
    "resnet152": (torchvision.models.resnet152, _resnet_backbone, _resnet_feats),
}


class ImageEmbedder(Task):
    """Task that classifies images.

    Args:
        embedding_dim: Dimension of the embedded vector. None uses the default from the backbone
        backbone: A model to use to extract image features.
        pretrained: Use a pretrained backbone.
        loss_fn: Loss function for training and finetuning, defaults to cross entropy.
        optimizer: Optimizer to use for training and finetuning, defaults to `torch.optim.SGD`.
        metrics: Metrics to compute for training and evaluation.
        learning_rate: Learning rate to use for training, defaults to `1e-3`

    Example::

        from flash.vision import ImageEmbedder

        embedder = ImageEmbedder(backbone='swav-imagenet')
        image = torch.rand(32, 3, 32, 32)
        embeddings = embedder(image)

    """

    def __init__(
        self,
        embedding_dim=None,
        backbone="swav-imagenet",
        pretrained=True,
        loss_fn: Callable = F.cross_entropy,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.SGD,
        metrics: Union[Callable, Mapping, Sequence, None] = (Accuracy()),
        learning_rate: float = 1e-3,
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

        if backbone in models:
            config = models[backbone]()
            self.backbone = config['model']
            num_features = config['num_features']

        elif backbone not in _backbones:
            raise NotImplementedError(f"Backbone {backbone} is not yet supported")

        else:
            backbone_fn, split, num_feats = _backbones[backbone]
            backbone = backbone_fn(pretrained=pretrained)
            self.backbone = split(backbone)
            num_features = num_feats(backbone)

        if embedding_dim is None:
            self.pooling = nn.Identity()
            self.head = nn.Identity()
        else:
            self.pooling = nn.AdaptiveAvgPool2d((1, 1)),
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_features, embedding_dim),
            )
            rank_zero_warn('embedding_dim is not None. Remember to finetune first!')

    def forward(self, x) -> Any:
        x = self.backbone(x)

        # bolts ssl models return lists
        if isinstance(x, tuple):
            x = x[-1]

        if len(x.size()) == 4 and self.embedding_dim is not None:
            x = self.pooling(x)

        x = self.head(x)
        x = x.view(x.size(0), -1)
        return x

    @staticmethod
    def default_pipeline() -> ImageClassificationDataPipeline:
        return ImageClassificationData.default_pipeline()


if __name__ == '__main__':
    embedder = ImageEmbedder(backbone='resnet50')
    image = torch.rand(32, 3, 128, 128)
    embeddings = embedder.predict('/Users/williamfalcon/Desktop/abcd.jpeg')
    print(embeddings.shape)
