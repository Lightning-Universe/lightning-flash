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
from typing import Any, Dict, List, Optional, Type, Union

from torch import nn, Tensor
from torch.nn import functional as F

from flash.core.classification import ClassificationTask
from flash.core.data.io.input import DataKeys, ServeInput
from flash.core.data.io.output import Output
from flash.core.data.io.output_transform import OutputTransform
from flash.core.model import Task
from flash.core.registry import FlashRegistry
from flash.core.serve import Composition
from flash.core.utilities.imports import (
    _TM_GREATER_EQUAL_0_7_0,
    _TM_GREATER_EQUAL_0_10_0,
    _TORCHVISION_AVAILABLE,
    _TORCHVISION_GREATER_EQUAL_0_9,
    requires,
)
from flash.core.utilities.isinstance import _isinstance
from flash.core.utilities.types import (
    INPUT_TRANSFORM_TYPE,
    LOSS_FN_TYPE,
    LR_SCHEDULER_TYPE,
    METRICS_TYPE,
    OPTIMIZER_TYPE,
    OUTPUT_TRANSFORM_TYPE,
)
from flash.image.data import ImageDeserializer
from flash.image.segmentation.backbones import SEMANTIC_SEGMENTATION_BACKBONES
from flash.image.segmentation.heads import SEMANTIC_SEGMENTATION_HEADS
from flash.image.segmentation.input_transform import SemanticSegmentationInputTransform
from flash.image.segmentation.output import SEMANTIC_SEGMENTATION_OUTPUTS

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as T

    if _TORCHVISION_GREATER_EQUAL_0_9:
        from torchvision.transforms import InterpolationMode
    else:

        class InterpolationMode:
            NEAREST = "nearest"


if _TM_GREATER_EQUAL_0_10_0:
    from torchmetrics.classification import MulticlassJaccardIndex as JaccardIndex
elif _TM_GREATER_EQUAL_0_7_0:
    from torchmetrics import JaccardIndex
else:
    from torchmetrics import IoU as JaccardIndex


class SemanticSegmentationOutputTransform(OutputTransform):
    def per_sample_transform(self, sample: Any) -> Any:
        resize = T.Resize(sample[DataKeys.METADATA]["size"], interpolation=InterpolationMode.NEAREST)
        sample[DataKeys.PREDS] = resize(sample[DataKeys.PREDS])
        sample[DataKeys.INPUT] = resize(sample[DataKeys.INPUT])
        return super().per_sample_transform(sample)


class SemanticSegmentation(ClassificationTask):
    """``SemanticSegmentation`` is a :class:`~flash.Task` for semantic segmentation of images. For more details, see
    :ref:`semantic_segmentation`.

    Args:
        num_classes: Number of classes to classify.
        backbone: A string or model to use to compute image features.
        backbone_kwargs: Additional arguments for the backbone configuration.
        head: A string or (model, num_features) tuple to use to compute image features.
        head_kwargs: Additional arguments for the head configuration.
        pretrained: Use a pretrained backbone.
        loss_fn: Loss function for training.
        optimizer: Optimizer to use for training.
        lr_scheduler: The LR scheduler to use during training.
        metrics: Metrics to compute for training and evaluation. Can either be an metric from the `torchmetrics`
            package, a custom metric inherenting from `torchmetrics.Metric`, a callable function or a list/dict
            containing a combination of the aforementioned. In all cases, each metric needs to have the signature
            `metric(preds,target)` and return a single scalar tensor. Defaults to :class:`torchmetrics.IOU`.
        learning_rate: Learning rate to use for training. If ``None`` (the default) then the default LR for your chosen
            optimizer will be used.
        multi_label: Whether the targets are multi-label or not.
        output: The :class:`~flash.core.data.io.output.Output` to use when formatting prediction outputs.
        output_transform: :class:`~flash.core.data.io.output_transform.OutputTransform` use for post processing samples.
    """

    output_transform_cls = SemanticSegmentationOutputTransform

    backbones: FlashRegistry = SEMANTIC_SEGMENTATION_BACKBONES
    heads: FlashRegistry = SEMANTIC_SEGMENTATION_HEADS
    outputs: FlashRegistry = Task.outputs + SEMANTIC_SEGMENTATION_OUTPUTS

    required_extras: str = "image"

    def __init__(
        self,
        num_classes: int,
        backbone: Union[str, nn.Module] = "resnet50",
        backbone_kwargs: Optional[Dict] = None,
        head: str = "fpn",
        head_kwargs: Optional[Dict] = None,
        pretrained: Union[bool, str] = True,
        loss_fn: LOSS_FN_TYPE = None,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        metrics: METRICS_TYPE = None,
        learning_rate: Optional[float] = None,
        multi_label: bool = False,
        output_transform: OUTPUT_TRANSFORM_TYPE = None,
    ) -> None:
        if metrics is None:
            metrics = JaccardIndex(num_classes=num_classes)

        if loss_fn is None:
            loss_fn = F.cross_entropy

        # TODO: need to check for multi_label
        if multi_label:
            raise NotImplementedError("Multi-label not supported yet.")

        super().__init__(
            model=None,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            learning_rate=learning_rate,
            output_transform=output_transform or self.output_transform_cls(),
        )

        self.save_hyperparameters()

        if not backbone_kwargs:
            backbone_kwargs = {}

        if not head_kwargs:
            head_kwargs = {}

        if isinstance(backbone, nn.Module):
            self.backbone = backbone
        else:
            self.backbone = self.backbones.get(backbone)(**backbone_kwargs)

        self.head: nn.Module = self.heads.get(head)(
            backbone=self.backbone, num_classes=num_classes, pretrained=pretrained, **head_kwargs
        )
        self.backbone = self.head.encoder

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DataKeys.INPUT], batch[DataKeys.TARGET])
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DataKeys.INPUT], batch[DataKeys.TARGET])
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DataKeys.INPUT], batch[DataKeys.TARGET])
        return super().test_step(batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        batch_input = batch[DataKeys.INPUT]
        batch[DataKeys.PREDS] = super().predict_step(batch_input, batch_idx, dataloader_idx=dataloader_idx)
        return batch

    def forward(self, x) -> Tensor:
        res = self.head(x)

        # some frameworks like torchvision return a dict.
        # In particular, torchvision segmentation models return the output logits
        # in the key `out`.
        if _isinstance(res, Dict[str, Tensor]):
            res = res["out"]

        return res

    @classmethod
    def available_pretrained_weights(cls, backbone: str):
        result = cls.backbones.get(backbone, with_metadata=True)
        pretrained_weights = None

        if "weights_paths" in result["metadata"]:
            pretrained_weights = list(result["metadata"]["weights_paths"])

        return pretrained_weights

    @requires("serve")
    def serve(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        sanity_check: bool = True,
        input_cls: Optional[Type[ServeInput]] = ImageDeserializer,
        transform: INPUT_TRANSFORM_TYPE = SemanticSegmentationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        output: Optional[Union[str, Output]] = None,
    ) -> Composition:
        return super().serve(host, port, sanity_check, input_cls, transform, transform_kwargs, output)

    @staticmethod
    def _ci_benchmark_fn(history: List[Dict[str, Any]]):
        """This function is used only for debugging usage with CI."""
        assert history[-1]["val_jaccardindex"] > 0.1
