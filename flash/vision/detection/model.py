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
from typing import Any, Callable, Mapping, Optional, Sequence, Type, Union

import torch
import torchvision
from torch import nn
from torch.optim import Optimizer
from torchvision.models.detection.faster_rcnn import FasterRCNN as torchvision_FasterRCNN
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import box_iou

from flash.core import Task
from flash.vision.backbones import fetch_fasterrcnn_backbone_and_num_features
from flash.vision.detection.data import ObjectDetectionDataPipeline
from flash.vision.detection.finetuning import ObjectDetectionFineTuning


def _evaluate_iou(target, pred):
    """
    Evaluate intersection over union (IOU) for target from dataset and output prediction from model
    """
    if pred["boxes"].shape[0] == 0:
        # no box detected, 0 IOU
        return torch.tensor(0.0, device=pred["boxes"].device)
    return box_iou(target["boxes"], pred["boxes"]).diag().mean()


class ObjectDetector(Task):
    """Object detection task

    Ref: Lightning Bolts https://github.com/PyTorchLightning/pytorch-lightning-bolts

    Args:
        num_classes: the number of classes for detection, including background
        backbone: Pretained backbone CNN architecture.
        fpn: If True, creates a Feature Pyramind Network on top of Resnet based CNNs.
        pretrained: if true, returns a model pre-trained on COCO train2017
        pretrained_backbone: if true, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers: number of trainable resnet layers starting from final block
        loss: the function(s) to update the model with. Has no effect for torchvision detection models.
        metrics: The provided metrics. All metrics here will be logged to progress bar and the respective logger.
        optimizer: The optimizer to use for training. Can either be the actual class or the class name.
        pretrained: Whether the model from torchvision should be loaded with it's pretrained weights.
            Has no effect for custom models.
        learning_rate: The learning rate to use for training

    """

    def __init__(
        self,
        num_classes: int,
        backbone: Optional[str] = None,
        fpn: bool = True,
        pretrained: bool = False,
        pretrained_backbone: bool = True,
        trainable_backbone_layers: int = 3,
        loss=None,
        metrics: Union[Callable, nn.Module, Mapping, Sequence, None] = None,
        optimizer: Type[Optimizer] = torch.optim.Adam,
        learning_rate: float = 1e-3,
        **kwargs: Any,
    ):

        self.save_hyperparameters()

        if backbone is None:
            model = fasterrcnn_resnet50_fpn(
                pretrained=pretrained,
                pretrained_backbone=pretrained_backbone,
                trainable_backbone_layers=trainable_backbone_layers,
            )
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            head = FastRCNNPredictor(in_features, num_classes)
            model.roi_heads.box_predictor = head
        else:
            backbone_model, num_features = fetch_fasterrcnn_backbone_and_num_features(
                backbone,
                fpn,
                pretrained_backbone,
                trainable_backbone_layers,
                **kwargs,
            )
            backbone_model.out_channels = num_features
            anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512), ), aspect_ratios=((0.5, 1.0, 2.0), ))
            model = torchvision_FasterRCNN(
                backbone_model, num_classes=num_classes, rpn_anchor_generator=anchor_generator, **kwargs
            )

        super().__init__(
            model=model,
            loss_fn=loss,
            metrics=metrics,
            learning_rate=learning_rate,
            optimizer=optimizer,
        )

    def training_step(self, batch, batch_idx) -> Any:
        """The training step. Overrides ``Task.training_step``
        """
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]

        # fasterrcnn takes both images and targets for training, returns loss_dict
        loss_dict = self.model(images, targets)
        loss = sum(loss_dict.values())
        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # fasterrcnn takes only images for eval() mode
        outs = self.model(images)
        iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()
        return {"val_iou": iou}

    def validation_epoch_end(self, outs):
        avg_iou = torch.stack([o["val_iou"] for o in outs]).mean()
        logs = {"val_iou": avg_iou}
        return {"avg_val_iou": avg_iou, "log": logs}

    def test_step(self, batch, batch_idx):
        images, targets = batch
        # fasterrcnn takes only images for eval() mode
        outs = self.model(images)
        iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()
        return {"test_iou": iou}

    def test_epoch_end(self, outs):
        avg_iou = torch.stack([o["test_iou"] for o in outs]).mean()
        logs = {"test_iou": avg_iou}
        return {"avg_test_iou": avg_iou, "log": logs}

    @staticmethod
    def default_pipeline() -> ObjectDetectionDataPipeline:
        return ObjectDetectionDataPipeline()

    def configure_finetune_callback(self):
        return [ObjectDetectionFineTuning(train_bn=True)]
