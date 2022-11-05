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
from types import FunctionType
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DistributedSampler
from torchmetrics import Accuracy

import flash
from flash.core.classification import ClassificationTask
from flash.core.data.io.input import DataKeys
from flash.core.registry import FlashRegistry
from flash.core.utilities.compatibility import accelerator_connector
from flash.core.utilities.imports import _PYTORCHVIDEO_AVAILABLE
from flash.core.utilities.providers import _PYTORCHVIDEO
from flash.core.utilities.types import LOSS_FN_TYPE, LR_SCHEDULER_TYPE, METRICS_TYPE, OPTIMIZER_TYPE

_VIDEO_CLASSIFIER_BACKBONES = FlashRegistry("backbones")

if _PYTORCHVIDEO_AVAILABLE:
    from pytorchvideo.models import hub

    for fn_name in dir(hub):
        if "__" not in fn_name:
            fn = getattr(hub, fn_name)
            if isinstance(fn, FunctionType):
                _VIDEO_CLASSIFIER_BACKBONES(fn=fn, providers=_PYTORCHVIDEO)


class VideoClassifier(ClassificationTask):
    """Task that classifies videos.

    Args:
        num_classes: Number of classes to classify.
        backbone: A string mapped to ``pytorch_video`` backbones or ``nn.Module``, defaults to ``"x3d_xs"``.
        backbone_kwargs: Arguments to customize the backbone from PyTorchVideo.
        pretrained: Use a pretrained backbone, defaults to ``True``.
        loss_fn: Loss function for training, defaults to :func:`torch.nn.functional.cross_entropy`.
        optimizer: Optimizer to use for training, defaults to :class:`torch.optim.SGD`.

        lr_scheduler: The scheduler or scheduler class to use.

        metrics: Metrics to compute for training and evaluation. Can either be an metric from the `torchmetrics`
            package, a custom metric inherenting from `torchmetrics.Metric`, a callable function or a list/dict
            containing a combination of the aforementioned. In all cases, each metric needs to have the signature
            `metric(preds,target)` and return a single scalar tensor. Defaults to :class:`torchmetrics.Accuracy`.
        learning_rate: Learning rate to use for training, defaults to ``1e-3``.
        head: either a `nn.Module` or a callable function that converts the features extrated from the backbone
            into class log probabilities (assuming default loss function). If `None`, will default to using
            a single linear layer.
    """

    backbones: FlashRegistry = _VIDEO_CLASSIFIER_BACKBONES

    required_extras = "video"

    def __init__(
        self,
        num_classes: Optional[int] = None,
        labels: Optional[List[str]] = None,
        backbone: Union[str, nn.Module] = "x3d_xs",
        backbone_kwargs: Optional[Dict] = None,
        pretrained: bool = True,
        loss_fn: LOSS_FN_TYPE = F.cross_entropy,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        metrics: METRICS_TYPE = Accuracy(),
        learning_rate: Optional[float] = None,
        head: Optional[Union[FunctionType, nn.Module]] = None,
    ):
        self.save_hyperparameters()

        if labels is not None and num_classes is None:
            num_classes = len(labels)

        super().__init__(
            model=None,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            learning_rate=learning_rate,
            num_classes=num_classes,
            labels=labels,
        )

        if not backbone_kwargs:
            backbone_kwargs = {}

        backbone_kwargs["pretrained"] = True if (flash._IS_TESTING and torch.cuda.is_available()) else pretrained
        backbone_kwargs["head_activation"] = None

        if isinstance(backbone, nn.Module):
            self.backbone = backbone
        elif isinstance(backbone, str):
            self.backbone = self.backbones.get(backbone)(**backbone_kwargs)
            num_features = self.backbone.blocks[-1].proj.out_features
        else:
            raise ValueError(f"backbone should be either a string or a nn.Module. Found: {backbone}")

        self.head = head or nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, num_classes),
        )

    def on_train_start(self) -> None:
        if accelerator_connector(self.trainer).is_distributed:
            encoded_dataset = self.trainer.train_dataloader.loaders.dataset.data
            encoded_dataset._video_sampler = DistributedSampler(encoded_dataset._labeled_videos)
        super().on_train_start()

    def on_train_epoch_start(self) -> None:
        if accelerator_connector(self.trainer).is_distributed:
            encoded_dataset = self.trainer.train_dataloader.loaders.dataset.data
            encoded_dataset._video_sampler.set_epoch(self.trainer.current_epoch)
        super().on_train_epoch_start()

    def step(self, batch: Any, batch_idx: int, metrics) -> Any:
        return super().step((batch[DataKeys.INPUT], batch[DataKeys.TARGET]), batch_idx, metrics)

    def forward(self, x: Any) -> Any:
        x = self.backbone(x)
        if self.head is not None:
            x = self.head(x)
        return x

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        predictions = self(batch[DataKeys.INPUT])
        batch[DataKeys.PREDS] = predictions
        return batch

    def modules_to_freeze(self) -> Union[nn.Module, Iterable[Union[nn.Module, Iterable]]]:
        """Return the module attributes of the model to be frozen."""
        return list(self.backbone.children())
