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
from typing import Any, Dict, List, Mapping, Optional, Type, Union

import torch
from pytorch_lightning.utilities import rank_zero_info
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from flash.core.adapter import AdapterTask
from flash.core.data.data_pipeline import DataPipeline
from flash.core.data.process import Serializer
from flash.core.data.serialization import Preds
from flash.core.registry import FlashRegistry
from flash.image.instance_segmentation.backbones import INSTANCE_SEGMENTATION_HEADS
from flash.image.instance_segmentation.data import InstanceSegmentationPostProcess, InstanceSegmentationPreprocess


class InstanceSegmentation(AdapterTask):
    """The ``InstanceSegmentation`` is a :class:`~flash.Task` for detecting objects in images. For more details, see
    :ref:`object_detection`.

    Args:
        num_classes: the number of classes for detection, including background
        backbone: Pretained backbone CNN architecture. Constructs a model with a
            ResNet-50-FPN backbone when no backbone is specified.
        head: string indicating the head module to use on top of the backbone
        pretrained: Whether the model from torchvision should be loaded with it's pretrained weights.
            Has no effect for custom models.
        optimizer: The optimizer to use for training. Can either be the actual class or the class name.
        optimizer_kwargs: Additional kwargs to use when creating the optimizer (if not passed as an instance).
        scheduler: The scheduler or scheduler class to use.
        scheduler_kwargs: Additional kwargs to use when creating the scheduler (if not passed as an instance).
        learning_rate: The learning rate to use for training
        serializer: A instance of :class:`~flash.core.data.process.Serializer` or a mapping consisting of such
            to use when serializing prediction outputs.
        **kwargs: additional kwargs used for initializing the task
    """

    heads: FlashRegistry = INSTANCE_SEGMENTATION_HEADS

    required_extras: List[str] = ["image", "icevision"]

    def __init__(
        self,
        num_classes: int,
        backbone: Optional[str] = "resnet18_fpn",
        head: Optional[str] = "mask_rcnn",
        pretrained: bool = True,
        optimizer: Type[Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Union[Type[_LRScheduler], str, _LRScheduler]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        learning_rate: float = 5e-4,
        serializer: Optional[Union[Serializer, Mapping[str, Serializer]]] = None,
        **kwargs: Any,
    ):
        self.save_hyperparameters()

        metadata = self.heads.get(head, with_metadata=True)
        adapter = metadata["metadata"]["adapter"].from_task(
            self,
            num_classes=num_classes,
            backbone=backbone,
            head=head,
            pretrained=pretrained,
            **kwargs,
        )

        super().__init__(
            adapter,
            learning_rate=learning_rate,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            serializer=serializer or Preds(),
        )

    def _ci_benchmark_fn(self, history: List[Dict[str, Any]]) -> None:
        """This function is used only for debugging usage with CI."""
        # todo

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        # todo: currently the data pipeline for icevision is not serializable, so we re-create the pipeline.
        if "data_pipeline" not in checkpoint:
            rank_zero_info(
                "Assigned Segmentation Data Pipeline for data processing. This is because a data-pipeline stored in "
                "the model due to pickling issues. "
                "If you'd like to change this, extend the InstanceSegmentation Task and override `on_load_checkpoint`."
            )
            self.data_pipeline = DataPipeline(
                preprocess=InstanceSegmentationPreprocess(), postprocess=InstanceSegmentationPostProcess()
            )
