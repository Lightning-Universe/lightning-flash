from typing import Callable, Mapping, Sequence, Union, Type

import torch
from torch import nn
import torch.nn.functional as F

from pl_flash import Model

import torchvision

_resnet_backbone = lambda model: nn.Sequential(*list(model.children())[:-2])
_resnet_feats = lambda model: model.fc.in_features

_backbones = {
    "resnet18": (torchvision.models.resnet18, _resnet_backbone, _resnet_feats),
    "resnet34": (torchvision.models.resnet34, _resnet_backbone, _resnet_feats),
    "resnet50": (torchvision.models.resnet50, _resnet_backbone, _resnet_feats),
    "resnet101": (torchvision.models.resnet101, _resnet_backbone, _resnet_feats),
    "resnet152": (torchvision.models.resnet152, _resnet_backbone, _resnet_feats),
}


class ImageClassifier(Model):
    def __init__(
        self,
        num_classes,
        backbone="resnet18",
        pretrained=True,
        loss_fn: Callable = F.cross_entropy,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.SGD,
        metrics: Union[Callable, Mapping, Sequence, None] = None,
        learning_rate: float = 1e-3,
    ):
        super().__init__(
            model=None,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,
        )

        if backbone not in _backbones:
            raise NotImplementedError(f"Backbone {backbone} is not yet supported")

        backbone_fn, split, num_feats = _backbones[backbone]
        backbone = backbone_fn(pretrained=pretrained)
        self.backbone = split(backbone)
        num_features = num_feats(backbone)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

    def freeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
