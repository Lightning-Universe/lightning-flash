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
from typing import Any, Dict, Optional, Tuple, Union

from torch import nn, Tensor
from torch.utils.data import DataLoader, Sampler

import flash
from flash.core.data.io.input import DataKeys, InputBase
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.utilities.collate import wrap_collate
from flash.core.model import Task
from flash.core.registry import FlashRegistry
from flash.core.utilities.apply_func import get_callable_dict
from flash.core.utilities.stability import beta
from flash.core.utilities.types import LOSS_FN_TYPE, LR_SCHEDULER_TYPE, METRICS_TYPE, OPTIMIZER_TYPE
from flash.pointcloud.detection.backbones import POINTCLOUD_OBJECT_DETECTION_BACKBONES

__FILE_EXAMPLE__ = "pointcloud_detection"


@beta("Point cloud object detection is currently in Beta.")
class PointCloudObjectDetector(Task):
    """The ``PointCloudObjectDetector`` is a :class:`~flash.core.classification.ClassificationTask` that classifies
    pointcloud data.

    Args:
        num_classes: The number of classes (outputs) for this :class:`~flash.core.model.Task`.
        backbone: The backbone name (or a tuple of ``nn.Module``, output size) to use.
        backbone_kwargs: Any additional kwargs to pass to the backbone constructor.
        loss_fn: The loss function to use. If ``None``, a default will be selected by the
            :class:`~flash.core.classification.ClassificationTask` depending on the ``multi_label`` argument.
        optimizer: Optimizer to use for training.
        lr_scheduler: The LR scheduler to use during training.
        metrics: Any metrics to use with this :class:`~flash.core.model.Task`. If ``None``, a default will be selected
            by the :class:`~flash.core.classification.ClassificationTask` depending on the ``multi_label`` argument.
        learning_rate: The learning rate for the optimizer.
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
        loss_fn: LOSS_FN_TYPE = None,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        metrics: METRICS_TYPE = None,
        learning_rate: Optional[float] = None,
        lambda_loss_cls: float = 1.0,
        lambda_loss_bbox: float = 1.0,
        lambda_loss_dir: float = 1.0,
    ):

        super().__init__(
            model=None,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            learning_rate=learning_rate,
        )

        self.save_hyperparameters()

        if backbone_kwargs is None:
            backbone_kwargs = {}

        if isinstance(backbone, tuple):
            self.backbone, out_features = backbone
        else:
            self.model, out_features, collate_fn = self.backbones.get(backbone)(**backbone_kwargs)
            self.collate_fn = wrap_collate(collate_fn)
            self.backbone = self.model.backbone
            self.neck = self.model.neck
            self.loss_fn = get_callable_dict(self.model.loss)

        if __FILE_EXAMPLE__ not in sys.argv[0]:
            self.model.bbox_head.conv_cls = self.head = nn.Conv2d(
                out_features, num_classes, kernel_size=(1, 1), stride=(1, 1)
            )

    def compute_loss(self, losses: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        losses = losses["loss"]
        return (
            self.hparams.lambda_loss_cls * losses["loss_cls"]
            + self.hparams.lambda_loss_bbox * losses["loss_bbox"]
            + self.hparams.lambda_loss_dir * losses["loss_dir"]
        )

    def compute_logs(self, logs: Dict[str, Any], losses: Dict[str, Tensor]):
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
            DataKeys.INPUT: getattr(batch, "point", None),
            DataKeys.PREDS: boxes,
            DataKeys.METADATA: [a["name"] for a in batch.attr],
        }

    def forward(self, x) -> Tensor:
        """First call the backbone, then the model head."""
        # hack to enable backbone to work properly.
        self.model.device = self.device
        return self.model(x)

    def _patch_dataset(self, dataset: InputBase):
        dataset.input_transform_fn = self.model.preprocess
        dataset.transform_fn = self.model.transform

    def process_train_dataset(
        self,
        dataset: InputBase,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = True,
        drop_last: bool = True,
        sampler: Optional[Sampler] = None,
        persistent_workers: bool = False,
        input_transform: Optional[InputTransform] = None,
        trainer: Optional["flash.Trainer"] = None,
    ) -> DataLoader:
        self._patch_dataset(dataset)
        return super().process_train_dataset(
            dataset,
            batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
            persistent_workers=persistent_workers,
            input_transform=input_transform,
            trainer=trainer,
        )

    def process_val_dataset(
        self,
        dataset: InputBase,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = False,
        drop_last: bool = False,
        sampler: Optional[Sampler] = None,
        persistent_workers: bool = False,
        input_transform: Optional[InputTransform] = None,
        trainer: Optional["flash.Trainer"] = None,
    ) -> DataLoader:
        self._patch_dataset(dataset)
        return super().process_val_dataset(
            dataset,
            batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
            persistent_workers=persistent_workers,
            input_transform=input_transform,
            trainer=trainer,
        )

    def process_test_dataset(
        self,
        dataset: InputBase,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = False,
        drop_last: bool = False,
        sampler: Optional[Sampler] = None,
        persistent_workers: bool = False,
        input_transform: Optional[InputTransform] = None,
        trainer: Optional["flash.Trainer"] = None,
    ) -> DataLoader:
        self._patch_dataset(dataset)
        return super().process_test_dataset(
            dataset,
            batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
            persistent_workers=persistent_workers,
            input_transform=input_transform,
            trainer=trainer,
        )

    def process_predict_dataset(
        self,
        dataset: InputBase,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = False,
        drop_last: bool = False,
        sampler: Optional[Sampler] = None,
        persistent_workers: bool = False,
        input_transform: Optional[InputTransform] = None,
        trainer: Optional["flash.Trainer"] = None,
    ) -> DataLoader:
        self._patch_dataset(dataset)
        return super().process_predict_dataset(
            dataset,
            batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
            persistent_workers=persistent_workers,
            input_transform=input_transform,
            trainer=trainer,
        )
