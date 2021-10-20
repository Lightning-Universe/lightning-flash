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
from typing import Any, Dict, List, Optional

from pytorch_lightning.utilities import rank_zero_info

from flash.core.adapter import AdapterTask
from flash.core.data.data_pipeline import DataPipeline
from flash.core.data.serialization import Preds
from flash.core.registry import FlashRegistry
from flash.core.utilities.types import LR_SCHEDULER_TYPE, OPTIMIZER_TYPE, SERIALIZER_TYPE
from flash.image.instance_segmentation.backbones import INSTANCE_SEGMENTATION_HEADS
from flash.image.instance_segmentation.data import InstanceSegmentationPostProcess, InstanceSegmentationPreprocess


class InstanceSegmentation(AdapterTask):
    """The ``InstanceSegmentation`` is a :class:`~flash.Task` for detecting objects in images. For more details, see
    :ref:`object_detection`.

    Args:
        num_classes: the number of classes for detection, including background
        model: a string of :attr`_models`. Defaults to 'fasterrcnn'.
        backbone: Pretained backbone CNN architecture. Constructs a model with a
            ResNet-50-FPN backbone when no backbone is specified.
        fpn: If True, creates a Feature Pyramind Network on top of Resnet based CNNs.
        pretrained: if true, returns a model pre-trained on COCO train2017
        pretrained_backbone: if true, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers: number of trainable resnet layers starting from final block.
            Only applicable for `fasterrcnn`.
        loss: the function(s) to update the model with. Has no effect for torchvision detection models.
        metrics: The provided metrics. All metrics here will be logged to progress bar and the respective logger.
            Changing this argument currently has no effect.
        optimizer: Optimizer to use for training.
        lr_scheduler: The LR scheduler to use during training.
        pretrained: Whether the model from torchvision should be loaded with it's pretrained weights.
            Has no effect for custom models.
        learning_rate: The learning rate to use for training

    """

    heads: FlashRegistry = INSTANCE_SEGMENTATION_HEADS

    required_extras: List[str] = ["image", "icevision"]

    def __init__(
        self,
        num_classes: int,
        backbone: Optional[str] = "resnet18_fpn",
        head: Optional[str] = "mask_rcnn",
        pretrained: bool = True,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        learning_rate: float = 5e-4,
        serializer: SERIALIZER_TYPE = None,
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
            lr_scheduler=lr_scheduler,
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
