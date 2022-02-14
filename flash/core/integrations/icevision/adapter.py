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
import functools
from typing import Any, Callable, Dict, List, Optional

from torch.utils.data import DataLoader, Sampler

import flash
from flash.core.adapter import Adapter
from flash.core.data.io.input import DataKeys, InputBase
from flash.core.integrations.icevision.transforms import (
    from_icevision_predictions,
    from_icevision_record,
    to_icevision_record,
)
from flash.core.model import Task
from flash.core.utilities.imports import _ICEVISION_AVAILABLE
from flash.core.utilities.url_error import catch_url_error

if _ICEVISION_AVAILABLE:
    from icevision.metrics import COCOMetric
    from icevision.metrics import Metric as IceVisionMetric
else:
    COCOMetric = object


class SimpleCOCOMetric(COCOMetric):
    def finalize(self) -> Dict[str, float]:
        logs = super().finalize()
        return {
            "Precision (IoU=0.50:0.95,area=all)": logs["AP (IoU=0.50:0.95) area=all"],
            "Recall (IoU=0.50:0.95,area=all,maxDets=100)": logs["AR (IoU=0.50:0.95) area=all maxDets=100"],
        }


class IceVisionAdapter(Adapter):
    """The ``IceVisionAdapter`` is an :class:`~flash.core.adapter.Adapter` for integrating with IceVision."""

    required_extras: str = "image"

    def __init__(self, model_type, model, icevision_adapter, backbone, predict_kwargs):
        super().__init__()

        self.model_type = model_type
        self.model = model
        self.icevision_adapter = icevision_adapter
        self.backbone = backbone
        self.predict_kwargs = predict_kwargs

    @classmethod
    @catch_url_error
    def from_task(
        cls,
        task: Task,
        num_classes: int,
        backbone: str,
        head: str,
        predict_kwargs: Dict,
        pretrained: bool = True,
        metrics: Optional["IceVisionMetric"] = None,
        image_size: Optional = None,
        **kwargs,
    ) -> Adapter:
        metadata = task.heads.get(head, with_metadata=True)
        backbones = metadata["metadata"]["backbones"]
        backbone_config = backbones.get(backbone)(pretrained)
        model_type, model, icevision_adapter, backbone = metadata["fn"](
            backbone_config,
            num_classes,
            image_size=image_size,
            **kwargs,
        )
        icevision_adapter = icevision_adapter(model=model, metrics=metrics)
        return cls(model_type, model, icevision_adapter, backbone, predict_kwargs)

    @staticmethod
    def _wrap_collate_fn(collate_fn, samples, metadata: Optional[List[Dict[str, Any]]] = None):
        metadata = metadata or [None] * len(samples)
        return {
            DataKeys.INPUT: collate_fn(
                [to_icevision_record({**sample, DataKeys.METADATA: m}) for sample, m in zip(samples, metadata)]
            ),
            DataKeys.METADATA: metadata,
        }

    def process_train_dataset(
        self,
        dataset: InputBase,
        trainer: "flash.Trainer",
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        collate_fn: Optional[Callable] = None,
        shuffle: bool = False,
        drop_last: bool = False,
        sampler: Optional[Sampler] = None,
        persistent_workers: bool = False,
    ) -> DataLoader:
        data_loader = self.model_type.train_dl(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
            persistent_workers=persistent_workers,
        )
        data_loader.collate_fn = functools.partial(self._wrap_collate_fn, data_loader.collate_fn)
        return data_loader

    def process_val_dataset(
        self,
        dataset: InputBase,
        trainer: "flash.Trainer",
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        collate_fn: Optional[Callable] = None,
        shuffle: bool = False,
        drop_last: bool = False,
        sampler: Optional[Sampler] = None,
        persistent_workers: bool = False,
    ) -> DataLoader:
        data_loader = self.model_type.valid_dl(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
            persistent_workers=persistent_workers,
        )
        data_loader.collate_fn = functools.partial(self._wrap_collate_fn, data_loader.collate_fn)
        return data_loader

    def process_test_dataset(
        self,
        dataset: InputBase,
        trainer: "flash.Trainer",
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        collate_fn: Optional[Callable] = None,
        shuffle: bool = False,
        drop_last: bool = False,
        sampler: Optional[Sampler] = None,
        persistent_workers: bool = False,
    ) -> DataLoader:
        data_loader = self.model_type.valid_dl(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
            persistent_workers=persistent_workers,
        )
        data_loader.collate_fn = functools.partial(self._wrap_collate_fn, data_loader.collate_fn)
        return data_loader

    def process_predict_dataset(
        self,
        dataset: InputBase,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        collate_fn: Callable = lambda x: x,
        shuffle: bool = False,
        drop_last: bool = True,
        sampler: Optional[Sampler] = None,
        persistent_workers: bool = False,
    ) -> DataLoader:
        data_loader = self.model_type.infer_dl(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
            persistent_workers=persistent_workers,
        )
        data_loader.collate_fn = functools.partial(self._wrap_collate_fn, data_loader.collate_fn)
        return data_loader

    def training_step(self, batch, batch_idx) -> Any:
        return self.icevision_adapter.training_step(batch[DataKeys.INPUT], batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.icevision_adapter.validation_step(batch[DataKeys.INPUT], batch_idx)

    def test_step(self, batch, batch_idx):
        return self.icevision_adapter.validation_step(batch[DataKeys.INPUT], batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        records = batch[DataKeys.INPUT][1]
        return {
            DataKeys.INPUT: [from_icevision_record(record) for record in records],
            DataKeys.PREDS: self(batch[DataKeys.INPUT]),
            DataKeys.METADATA: batch[DataKeys.METADATA],
        }

    def forward(self, batch: Any) -> Any:
        return from_icevision_predictions(
            self.model_type.predict_from_dl(self.model, [batch], show_pbar=False, **self.predict_kwargs)
        )

    def training_epoch_end(self, outputs) -> None:
        return self.icevision_adapter.training_epoch_end(outputs)

    def validation_epoch_end(self, outputs) -> None:
        return self.icevision_adapter.validation_epoch_end(outputs)

    def test_epoch_end(self, outputs) -> None:
        return self.icevision_adapter.validation_epoch_end(outputs)
