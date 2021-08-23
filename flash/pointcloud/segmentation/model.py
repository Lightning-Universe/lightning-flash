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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import torch
import torchmetrics
from pytorch_lightning import Callback, LightningModule
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Sampler
from torchmetrics import IoU

from flash.core.classification import ClassificationTask
from flash.core.data.auto_dataset import BaseAutoDataset
from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.process import Serializer
from flash.core.data.states import CollateFn
from flash.core.finetuning import BaseFinetuning
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _POINTCLOUD_AVAILABLE
from flash.pointcloud.segmentation.backbones import POINTCLOUD_SEGMENTATION_BACKBONES

if _POINTCLOUD_AVAILABLE:
    from open3d._ml3d.torch.modules.losses.semseg_loss import filter_valid_label
    from open3d.ml.torch.dataloaders import TorchDataloader


class PointCloudSegmentationFinetuning(BaseFinetuning):
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


class PointCloudSegmentationSerializer(Serializer):
    pass


class PointCloudSegmentation(ClassificationTask):
    """The ``PointCloudClassifier`` is a :class:`~flash.core.classification.ClassificationTask` that classifies
    pointcloud data.

    Args:
        num_features: The number of features (elements) in the input data.
        num_classes: The number of classes (outputs) for this :class:`~flash.core.model.Task`.
        backbone: The backbone name (or a tuple of ``nn.Module``, output size) to use.
        backbone_kwargs: Any additional kwargs to pass to the backbone constructor.
        loss_fn: The loss function to use. If ``None``, a default will be selected by the
            :class:`~flash.core.classification.ClassificationTask` depending on the ``multi_label`` argument.
        optimizer: The optimizer or optimizer class to use.
        optimizer_kwargs: Additional kwargs to use when creating the optimizer (if not passed as an instance).
        scheduler: The scheduler or scheduler class to use.
        scheduler_kwargs: Additional kwargs to use when creating the scheduler (if not passed as an instance).
        metrics: Any metrics to use with this :class:`~flash.core.model.Task`. If ``None``, a default will be selected
            by the :class:`~flash.core.classification.ClassificationTask` depending on the ``multi_label`` argument.
        learning_rate: The learning rate for the optimizer.
        multi_label: If ``True``, this will be treated as a multi-label classification problem.
        serializer: The :class:`~flash.core.data.process.Serializer` to use for prediction outputs.
    """

    backbones: FlashRegistry = POINTCLOUD_SEGMENTATION_BACKBONES

    required_extras: str = "pointcloud"

    def __init__(
        self,
        num_classes: int,
        backbone: Union[str, Tuple[nn.Module, int]] = "RandLANet",
        backbone_kwargs: Optional[Dict] = None,
        head: Optional[nn.Module] = None,
        loss_fn: Optional[Callable] = torch.nn.functional.cross_entropy,
        optimizer: Union[Type[torch.optim.Optimizer], torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Union[Type[_LRScheduler], str, _LRScheduler]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        metrics: Union[torchmetrics.Metric, Mapping, Sequence, None] = None,
        learning_rate: float = 1e-2,
        multi_label: bool = False,
        serializer: Optional[Union[Serializer, Mapping[str, Serializer]]] = PointCloudSegmentationSerializer(),
    ):
        import flash

        if metrics is None:
            metrics = IoU(num_classes=num_classes)

        super().__init__(
            model=None,
            loss_fn=loss_fn,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            metrics=metrics,
            learning_rate=learning_rate,
            multi_label=multi_label,
            serializer=serializer,
        )

        self.save_hyperparameters()

        if not backbone_kwargs:
            backbone_kwargs = {"num_classes": num_classes}

        if isinstance(backbone, tuple):
            self.backbone, out_features = backbone
        else:
            self.backbone, out_features, collate_fn = self.backbones.get(backbone)(**backbone_kwargs)
            # replace latest layer
            if not flash._IS_TESTING:
                self.backbone.fc = nn.Identity()
            self.set_state(CollateFn(collate_fn))

        self.head = nn.Identity() if flash._IS_TESTING else (head or nn.Linear(out_features, num_classes))

    def apply_filtering(self, labels, scores):
        scores, labels = filter_valid_label(scores, labels, self.hparams.num_classes, [0], self.device)
        return labels, scores

    def to_metrics_format(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.to_loss_format(x), dim=-1)

    def to_loss_format(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, x.shape[-1])

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.INPUT]["labels"].view(-1))
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.INPUT]["labels"].view(-1))
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.INPUT]["labels"].view(-1))
        return super().test_step(batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        batch[DefaultDataKeys.PREDS] = self(batch[DefaultDataKeys.INPUT])
        batch[DefaultDataKeys.TARGET] = batch[DefaultDataKeys.INPUT]["labels"]
        # drop sub-sampled pointclouds
        batch[DefaultDataKeys.INPUT] = batch[DefaultDataKeys.INPUT]["xyz"][0]
        return batch

    def forward(self, x) -> torch.Tensor:
        """First call the backbone, then the model head."""
        # hack to enable backbone to work properly.
        self.backbone.device = self.device
        x = self.backbone(x)
        if self.head is not None:
            x = self.head(x)
        return x

    def _process_dataset(
        self,
        dataset: BaseAutoDataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        collate_fn: Callable,
        shuffle: bool = False,
        drop_last: bool = True,
        sampler: Optional[Sampler] = None,
    ) -> DataLoader:

        if not _POINTCLOUD_AVAILABLE:
            raise ModuleNotFoundError("Please, run `pip install flash[pointcloud]`.")

        if not isinstance(dataset.dataset, TorchDataloader):

            dataset.dataset = TorchDataloader(
                dataset.dataset,
                preprocess=self.backbone.preprocess,
                transform=self.backbone.transform,
                use_cache=False,
            )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
        )

    def configure_finetune_callback(self) -> List[Callback]:
        return [PointCloudSegmentationFinetuning()]
