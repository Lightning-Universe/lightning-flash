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
from torch import nn
from torch.nn import functional as F
from torchmetrics import IoU

from flash.core.classification import ClassificationTask
from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.process import Postprocess, Serializer
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _KORNIA_AVAILABLE, _TIMM_AVAILABLE, _TORCHVISION_AVAILABLE
from flash.image.segmentation.backbones import SEMANTIC_SEGMENTATION_BACKBONES
from flash.image.segmentation.serialization import SegmentationLabels

if _KORNIA_AVAILABLE:
    import kornia as K


class SemanticSegmentationPostprocess(Postprocess):

    def per_sample_transform(self, sample: Any) -> Any:
        resize = K.geometry.Resize(sample[DefaultDataKeys.METADATA][-2:], interpolation='bilinear')
        sample[DefaultDataKeys.PREDS] = resize(torch.stack(sample[DefaultDataKeys.PREDS]))
        sample[DefaultDataKeys.INPUT] = resize(torch.stack(sample[DefaultDataKeys.INPUT]))
        return super().per_sample_transform(sample)


class SemanticSegmentation(ClassificationTask):
    """Task that performs semantic segmentation on images.

    Use a built in backbone

    Example::

        from flash.image import SemanticSegmentation

        segmentation = SemanticSegmentation(
            num_classes=21, backbone="torchvision/fcn_resnet50"
        )

    Args:
        num_classes: Number of classes to classify.
        backbone: A string or (model, num_features) tuple to use to compute image features,
            defaults to ``"torchvision/fcn_resnet50"``.
        backbone_kwargs: Additional arguments for the backbone configuration.
        pretrained: Use a pretrained backbone, defaults to ``False``.
        loss_fn: Loss function for training, defaults to :func:`torch.nn.functional.cross_entropy`.
        optimizer: Optimizer to use for training, defaults to :class:`torch.optim.AdamW`.
        metrics: Metrics to compute for training and evaluation, defaults to :class:`torchmetrics.IoU`.
        learning_rate: Learning rate to use for training, defaults to ``1e-3``.
        multi_label: Whether the targets are multi-label or not.
        serializer: The :class:`~flash.core.data.process.Serializer` to use when serializing prediction outputs.
    """

    postprocess_cls = SemanticSegmentationPostprocess

    backbones: FlashRegistry = SEMANTIC_SEGMENTATION_BACKBONES

    def __init__(
        self,
        num_classes: int,
        backbone: Union[str, Tuple[nn.Module, int]] = "torchvision/fcn_resnet50",
        backbone_kwargs: Optional[Dict] = None,
        pretrained: bool = True,
        loss_fn: Optional[Callable] = None,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.AdamW,
        metrics: Optional[Union[Callable, Mapping, Sequence, None]] = None,
        learning_rate: float = 1e-3,
        multi_label: bool = False,
        serializer: Optional[Union[Serializer, Mapping[str, Serializer]]] = None,
        postprocess: Optional[Postprocess] = None,
    ) -> None:

        if isinstance(backbone, str) and (not _TORCHVISION_AVAILABLE or not _TIMM_AVAILABLE):
            raise ModuleNotFoundError("Please, pip install -e '.[image]'")

        if metrics is None:
            metrics = IoU(num_classes=num_classes)

        if loss_fn is None:
            loss_fn = F.cross_entropy

        # TODO: need to check for multi_label
        if multi_label:
            raise NotImplementedError("Multi-label not supported yet.")

        super().__init__(
            model=None,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,
            serializer=serializer or SegmentationLabels(),
            postprocess=postprocess or self.postprocess_cls()
        )

        self.save_hyperparameters()

        if not backbone_kwargs:
            backbone_kwargs = {}

        # TODO: pretrained to True causes some issues
        self.backbone = self.backbones.get(backbone)(num_classes, pretrained=pretrained, **backbone_kwargs)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return super().test_step(batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        batch_input = (batch[DefaultDataKeys.INPUT])
        preds = super().predict_step(batch_input, batch_idx, dataloader_idx=dataloader_idx)
        batch[DefaultDataKeys.PREDS] = preds
        return batch

    def forward(self, x) -> torch.Tensor:
        # infer the image to the model
        res: Union[torch.Tensor, Dict[str, torch.Tensor]] = self.backbone(x)

        # some frameworks like torchvision return a dict.
        # In particular, torchvision segmentation models return the output logits
        # in the key `out`.
        out: torch.Tensor
        if isinstance(res, dict):
            out = res['out']
        else:
            raise NotImplementedError(f"Unsupported output type: {type(out)}")

        return out

    def _ci_benchmark_fn(self, history: List[Dict[str, Any]]):
        """
        This function is used only for debugging usage with CI
        """
        assert history[-1]["val_iou"] > 0.2
