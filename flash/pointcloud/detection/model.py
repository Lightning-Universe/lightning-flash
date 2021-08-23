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
import sys
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Type, Union

import torch
import torchmetrics
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Sampler

from flash.core.data.auto_dataset import BaseAutoDataset
from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.process import Serializer
from flash.core.data.states import CollateFn
from flash.core.model import Task
from flash.core.registry import FlashRegistry
from flash.core.utilities.apply_func import get_callable_dict
from flash.core.utilities.imports import _POINTCLOUD_AVAILABLE
from flash.pointcloud.detection.backbones import POINTCLOUD_OBJECT_DETECTION_BACKBONES

__FILE_EXAMPLE__ = "pointcloud_detection"


class PointCloudObjectDetectorSerializer(Serializer):
    pass


class PointCloudObjectDetector(Task):
    """The ``PointCloudObjectDetector`` is a :class:`~flash.core.classification.ClassificationTask` that classifies
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
        lambda_loss_cls: The value to scale the loss classification.
        lambda_loss_bbox: The value to scale the bounding boxes loss.
        lambda_loss_dir: The value to scale the bounding boxes direction loss.
    """

    backbones: FlashRegistry = POINTCLOUD_OBJECT_DETECTION_BACKBONES
    required_extras: str = "pointcloud"

    def __init__(
        self,
        num_classes: int,
        backbone: Union[str, Tuple[nn.Module, int]] = "pointpillars_kitti",
        backbone_kwargs: Optional[Dict] = None,
        head: Optional[nn.Module] = None,
        loss_fn: Optional[Callable] = None,
        optimizer: Union[Type[torch.optim.Optimizer], torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Union[Type[_LRScheduler], str, _LRScheduler]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        metrics: Union[torchmetrics.Metric, Mapping, Sequence, None] = None,
        learning_rate: float = 1e-2,
        serializer: Optional[Union[Serializer, Mapping[str, Serializer]]] = PointCloudObjectDetectorSerializer(),
        lambda_loss_cls: float = 1.0,
        lambda_loss_bbox: float = 1.0,
        lambda_loss_dir: float = 1.0,
    ):

        super().__init__(
            model=None,
            loss_fn=loss_fn,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            metrics=metrics,
            learning_rate=learning_rate,
            serializer=serializer,
        )

        self.save_hyperparameters()

        if backbone_kwargs is None:
            backbone_kwargs = {}

        if isinstance(backbone, tuple):
            self.backbone, out_features = backbone
        else:
            self.model, out_features, collate_fn = self.backbones.get(backbone)(**backbone_kwargs)
            self.backbone = self.model.backbone
            self.neck = self.model.neck
            self.set_state(CollateFn(collate_fn))
            self.set_state(CollateFn(collate_fn))
            self.set_state(CollateFn(collate_fn))
            self.loss_fn = get_callable_dict(self.model.loss)

        if __FILE_EXAMPLE__ not in sys.argv[0]:
            self.model.bbox_head.conv_cls = self.head = nn.Conv2d(
                out_features, num_classes, kernel_size=(1, 1), stride=(1, 1)
            )

    def compute_loss(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        losses = losses["loss"]
        return (
            self.hparams.lambda_loss_cls * losses["loss_cls"]
            + self.hparams.lambda_loss_bbox * losses["loss_bbox"]
            + self.hparams.lambda_loss_dir * losses["loss_dir"]
        )

    def compute_logs(self, logs: Dict[str, Any], losses: Dict[str, torch.Tensor]):
        logs.update({"loss": self.compute_loss(losses)})
        return logs

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        return super().training_step((batch, batch), batch_idx)

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        super().validation_step((batch, batch), batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        super().validation_step((batch, batch), batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        results = self.model(batch)
        boxes = self.model.inference_end(results, batch)
        return {
            DefaultDataKeys.INPUT: getattr(batch, "point", None),
            DefaultDataKeys.PREDS: boxes,
            DefaultDataKeys.METADATA: [a["name"] for a in batch.attr],
        }

    def forward(self, x) -> torch.Tensor:
        """First call the backbone, then the model head."""
        # hack to enable backbone to work properly.
        self.model.device = self.device
        return self.model(x)

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

        dataset.preprocess_fn = self.model.preprocess
        dataset.transform_fn = self.model.transform

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
