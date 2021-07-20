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
from typing import Any, Callable, Dict, List, Mapping, Optional, Type, Union

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Sampler

from flash.core.data.auto_dataset import BaseAutoDataset
from flash.core.data.process import Deserializer, Postprocess, Preprocess, Serializer
from flash.core.model import Task
from flash.core.utilities.imports import _ICEVISION_AVAILABLE

if _ICEVISION_AVAILABLE:
    from icevision.core import BaseRecord
    from icevision.data import Dataset
    from icevision.metrics import COCOMetric
    from icevision.metrics import Metric as IceVisionMetric


class SimpleCOCOMetric(COCOMetric):

    def finalize(self) -> Dict[str, float]:
        logs = super().finalize()
        return {
            "Precision (IoU=0.50:0.95,area=all)": logs["AP (IoU=0.50:0.95) area=all"],
            "Recall (IoU=0.50:0.95,area=all,maxDets=100)": logs["AR (IoU=0.50:0.95) area=all maxDets=100"],
        }


class IceVisionTask(Task):
    """The ``IceVisionTask`` is a base :class:`~flash.Task` for integrating with IceVision.

    Args:
        num_classes: the number of classes for detection, including background
        model: a string of :attr`_models`. Defaults to 'fasterrcnn'.
        backbone: Pretrained backbone CNN architecture. Constructs a model with a
            ResNet-50-FPN backbone when no backbone is specified.
        pretrained: if true, returns a model pre-trained on COCO train2017.
        metrics: The provided metrics. All metrics here will be logged to progress bar and the respective logger.
        image_size
    """

    required_extras: str = "image"

    def __init__(
        self,
        num_classes: int,
        backbone: str,
        head: str,
        pretrained: bool = True,
        optimizer: Union[Type[torch.optim.Optimizer], torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Union[Type[_LRScheduler], str, _LRScheduler]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        metrics: Optional[IceVisionMetric] = None,
        learning_rate: float = 5e-4,
        deserializer: Optional[Union[Deserializer, Mapping[str, Deserializer]]] = None,
        preprocess: Optional[Preprocess] = None,
        postprocess: Optional[Postprocess] = None,
        serializer: Optional[Union[Serializer, Mapping[str, Serializer]]] = None,
        image_size: Optional = None,
        **kwargs,
    ):
        self.save_hyperparameters()

        super().__init__(
            model=None,
            metrics=None,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            learning_rate=learning_rate,
            deserializer=deserializer,
            preprocess=preprocess,
            postprocess=postprocess,
            serializer=serializer,
        )

        metadata = self.heads.get(head, with_metadata=True)
        backbones = metadata["metadata"]["backbones"]
        backbone_config = backbones.get(backbone)(pretrained)
        self.model_type, self.model, adapter, self.backbone = metadata["fn"](
            backbone_config,
            num_classes,
            image_size=image_size,
            **kwargs,
        )
        self.adapter = adapter(model=self.model, metrics=metrics)

    @classmethod
    def available_backbones(cls, head: str) -> List[str]:
        metadata = cls.heads.get(head, with_metadata=True)
        backbones = metadata["metadata"]["backbones"]
        return backbones.available_keys()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self._data_pipeline_state is not None and '_data_pipeline_state' not in checkpoint:
            checkpoint['_data_pipeline_state'] = self._data_pipeline_state

    def process_train_dataset(
        self,
        dataset: BaseAutoDataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        collate_fn: Callable,
        shuffle: bool = False,
        drop_last: bool = False,
        sampler: Optional[Sampler] = None
    ) -> DataLoader:
        return self.model_type.train_dl(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
        )

    def process_val_dataset(
        self,
        dataset: BaseAutoDataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        collate_fn: Callable,
        shuffle: bool = False,
        drop_last: bool = False,
        sampler: Optional[Sampler] = None
    ) -> DataLoader:
        return self.model_type.valid_dl(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
        )

    def process_test_dataset(
        self,
        dataset: BaseAutoDataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        collate_fn: Callable,
        shuffle: bool = False,
        drop_last: bool = False,
        sampler: Optional[Sampler] = None
    ) -> DataLoader:
        return self.model_type.valid_dl(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
        )

    def process_predict_dataset(
        self,
        dataset: BaseAutoDataset,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        collate_fn: Callable = lambda x: x,
        shuffle: bool = False,
        drop_last: bool = True,
        sampler: Optional[Sampler] = None,
        convert_to_dataloader: bool = True
    ) -> Union[DataLoader, BaseAutoDataset]:
        if convert_to_dataloader:
            return self.model_type.infer_dl(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                shuffle=shuffle,
                drop_last=drop_last,
                sampler=sampler,
            )
        return dataset

    def training_step(self, batch, batch_idx) -> Any:
        return self.adapter.training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.adapter.validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.adapter.validation_step(batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if isinstance(batch, list) and isinstance(batch[0], BaseRecord):
            data = Dataset(batch)
            return self.model_type.predict(self.model, data)
        return self.model_type.predict_from_dl(self.model, [batch], show_pbar=False)

    def training_epoch_end(self, outputs) -> None:
        return self.adapter.training_epoch_end(outputs)

    def validation_epoch_end(self, outputs) -> None:
        return self.adapter.validation_epoch_end(outputs)

    def test_epoch_end(self, outputs) -> None:
        return self.adapter.validation_epoch_end(outputs)
