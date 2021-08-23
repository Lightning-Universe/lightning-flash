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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Type, Union

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler
from torchmetrics import Accuracy, Metric

import flash
from flash.core.classification import ClassificationTask, Labels
from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.process import Serializer
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _PYTORCHVIDEO_AVAILABLE
from flash.core.utilities.providers import _PYTORCHVIDEO

_VIDEO_CLASSIFIER_BACKBONES = FlashRegistry("backbones")

if _PYTORCHVIDEO_AVAILABLE:
    from pytorchvideo.models import hub

    for fn_name in dir(hub):
        if "__" not in fn_name:
            fn = getattr(hub, fn_name)
            if isinstance(fn, FunctionType):
                _VIDEO_CLASSIFIER_BACKBONES(fn=fn, providers=_PYTORCHVIDEO)


class VideoClassifierFinetuning(BaseFinetuning):
    def __init__(self, num_layers: int = 5, train_bn: bool = True, unfreeze_epoch: int = 1):
        super().__init__()
        self.num_layers = num_layers
        self.train_bn = train_bn
        self.unfreeze_epoch = unfreeze_epoch

    def freeze_before_training(self, pl_module: LightningModule) -> None:
        self.freeze(modules=list(pl_module.backbone.children())[: -self.num_layers], train_bn=self.train_bn)

    def finetune_function(
        self,
        pl_module: LightningModule,
        epoch: int,
        optimizer: Optimizer,
        opt_idx: int,
    ) -> None:
        if epoch != self.unfreeze_epoch:
            return
        self.unfreeze_and_add_param_group(
            modules=list(pl_module.backbone.children())[-self.num_layers :],
            optimizer=optimizer,
            train_bn=self.train_bn,
        )


class VideoClassifier(ClassificationTask):
    """Task that classifies videos.

    Args:
        num_classes: Number of classes to classify.
        backbone: A string mapped to ``pytorch_video`` backbones or ``nn.Module``, defaults to ``"slowfast_r50"``.
        backbone_kwargs: Arguments to customize the backbone from PyTorchVideo.
        pretrained: Use a pretrained backbone, defaults to ``True``.
        loss_fn: Loss function for training, defaults to :func:`torch.nn.functional.cross_entropy`.
        optimizer: Optimizer to use for training, defaults to :class:`torch.optim.SGD`.
        metrics: Metrics to compute for training and evaluation. Can either be an metric from the `torchmetrics`
            package, a custom metric inherenting from `torchmetrics.Metric`, a callable function or a list/dict
            containing a combination of the aforementioned. In all cases, each metric needs to have the signature
            `metric(preds,target)` and return a single scalar tensor. Defaults to :class:`torchmetrics.Accuracy`.
        learning_rate: Learning rate to use for training, defaults to ``1e-3``.
    """

    backbones: FlashRegistry = _VIDEO_CLASSIFIER_BACKBONES

    required_extras = "video"

    def __init__(
        self,
        num_classes: int,
        backbone: Union[str, nn.Module] = "x3d_xs",
        backbone_kwargs: Optional[Dict] = None,
        pretrained: bool = True,
        loss_fn: Callable = F.cross_entropy,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.SGD,
        metrics: Union[Metric, Callable, Mapping, Sequence, None] = Accuracy(),
        learning_rate: float = 1e-3,
        head: Optional[Union[FunctionType, nn.Module]] = None,
        serializer: Optional[Serializer] = None,
    ):
        super().__init__(
            model=None,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,
            serializer=serializer or Labels(),
        )

        self.save_hyperparameters()

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
            raise MisconfigurationException(f"backbone should be either a string or a nn.Module. Found: {backbone}")

        self.head = head or nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, num_classes),
        )

    def on_train_start(self) -> None:
        if self.trainer.accelerator_connector.is_distributed:
            encoded_dataset = self.trainer.train_dataloader.loaders.dataset.dataset
            encoded_dataset._video_sampler = DistributedSampler(encoded_dataset._labeled_videos)
        super().on_train_start()

    def on_train_epoch_start(self) -> None:
        if self.trainer.accelerator_connector.is_distributed:
            encoded_dataset = self.trainer.train_dataloader.loaders.dataset.dataset
            encoded_dataset._video_sampler.set_epoch(self.trainer.current_epoch)
        super().on_train_epoch_start()

    def step(self, batch: Any, batch_idx: int, metrics) -> Any:
        return super().step((batch["video"], batch["label"]), batch_idx, metrics)

    def forward(self, x: Any) -> Any:
        x = self.backbone(x)
        if self.head is not None:
            x = self.head(x)
        return x

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        predictions = self(batch["video"])
        batch[DefaultDataKeys.PREDS] = predictions
        return batch

    def configure_finetune_callback(self) -> List[Callback]:
        return [VideoClassifierFinetuning()]

    @staticmethod
    def _ci_benchmark_fn(history: List[Dict[str, Any]]):
        """This function is used only for debugging usage with CI."""
        assert history[-1]["val_accuracy"] > 0.70
