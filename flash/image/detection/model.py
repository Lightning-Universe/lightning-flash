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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Type, Union

import torch
from torch import nn, tensor
from torch.optim import Optimizer

from flash.core.data.data_source import DefaultDataKeys
from flash.core.model import Task
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _IMAGE_AVAILABLE
from flash.image.backbones import OBJ_DETECTION_BACKBONES
from flash.image.detection.finetuning import ObjectDetectionFineTuning

if _IMAGE_AVAILABLE:
    import torchvision
    from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
    from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead
    from torchvision.models.detection.rpn import AnchorGenerator
    from torchvision.ops import box_iou

    _models = {
        "fasterrcnn": torchvision.models.detection.fasterrcnn_resnet50_fpn,
        "retinanet": torchvision.models.detection.retinanet_resnet50_fpn,
    }

else:
    AnchorGenerator = None


def _evaluate_iou(target, pred):
    """
    Evaluate intersection over union (IOU) for target from dataset and output prediction from model
    """
    if pred["boxes"].shape[0] == 0:
        # no box detected, 0 IOU
        return tensor(0.0, device=pred["boxes"].device)
    return box_iou(target["boxes"], pred["boxes"]).diag().mean()


class ObjectDetector(Task):
    """Object detection task

    Ref: Lightning Bolts https://github.com/PyTorchLightning/lightning-bolts

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
        optimizer: The optimizer to use for training. Can either be the actual class or the class name.
        pretrained: Whether the model from torchvision should be loaded with it's pretrained weights.
            Has no effect for custom models.
        learning_rate: The learning rate to use for training

    """

    backbones: FlashRegistry = OBJ_DETECTION_BACKBONES

    def __init__(
        self,
        num_classes: int,
        model: str = "fasterrcnn",
        backbone: Optional[str] = None,
        fpn: bool = True,
        pretrained: bool = True,
        pretrained_backbone: bool = True,
        trainable_backbone_layers: int = 3,
        anchor_generator: Optional[Type['AnchorGenerator']] = None,
        loss=None,
        metrics: Union[Callable, nn.Module, Mapping, Sequence, None] = None,
        optimizer: Type[Optimizer] = torch.optim.AdamW,
        learning_rate: float = 1e-3,
        **kwargs: Any,
    ):

        if not _IMAGE_AVAILABLE:
            raise ModuleNotFoundError("Please, pip install . '[image]'")

        self.save_hyperparameters()

        if model in _models:
            model = ObjectDetector.get_model(
                model, num_classes, backbone, fpn, pretrained, pretrained_backbone, trainable_backbone_layers,
                anchor_generator, **kwargs
            )
        else:
            ValueError(f"{model} is not supported yet.")

        super().__init__(
            model=model,
            loss_fn=loss,
            metrics=metrics,
            learning_rate=learning_rate,
            optimizer=optimizer,
        )

    @staticmethod
    def get_model(
        model_name,
        num_classes,
        backbone,
        fpn,
        pretrained,
        pretrained_backbone,
        trainable_backbone_layers,
        anchor_generator,
        **kwargs,
    ):
        if backbone is None:
            # Constructs a model with a ResNet-50-FPN backbone when no backbone is specified.
            if model_name == "fasterrcnn":
                model = _models[model_name](
                    pretrained=pretrained,
                    pretrained_backbone=pretrained_backbone,
                    trainable_backbone_layers=trainable_backbone_layers,
                )
                in_features = model.roi_heads.box_predictor.cls_score.in_features
                head = FastRCNNPredictor(in_features, num_classes)
                model.roi_heads.box_predictor = head
            else:
                model = _models[model_name](pretrained=pretrained, pretrained_backbone=pretrained_backbone)
                model.head = RetinaNetHead(
                    in_channels=model.backbone.out_channels,
                    num_anchors=model.head.classification_head.num_anchors,
                    num_classes=num_classes,
                    **kwargs
                )
        else:
            backbone_model, num_features = ObjectDetector.backbones.get(backbone)(
                pretrained=pretrained_backbone,
                trainable_layers=trainable_backbone_layers,
                **kwargs,
            )
            backbone_model.out_channels = num_features
            if anchor_generator is None:
                anchor_generator = AnchorGenerator(
                    sizes=((32, 64, 128, 256, 512), ), aspect_ratios=((0.5, 1.0, 2.0), )
                ) if not hasattr(backbone_model, "fpn") else None

            if model_name == "fasterrcnn":
                model = FasterRCNN(backbone_model, num_classes=num_classes, rpn_anchor_generator=anchor_generator)
            else:
                model = RetinaNet(backbone_model, num_classes=num_classes, anchor_generator=anchor_generator)
        return model

    def training_step(self, batch, batch_idx) -> Any:
        """The training step. Overrides ``Task.training_step``
        """
        images, targets = batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET]
        targets = [{k: v for k, v in t.items()} for t in targets]

        # fasterrcnn takes both images and targets for training, returns loss_dict
        loss_dict = self.model(images, targets)
        loss = sum(loss_dict.values())
        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET]
        # fasterrcnn takes only images for eval() mode
        outs = self.model(images)
        iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()
        self.log("val_iou", iou)

    def on_validation_end(self) -> None:
        return super().on_validation_end()

    def test_step(self, batch, batch_idx):
        images, targets = batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET]
        # fasterrcnn takes only images for eval() mode
        outs = self.model(images)
        iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()
        self.log("test_iou", iou)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        images = batch[DefaultDataKeys.INPUT]
        return self.model(images)

    def configure_finetune_callback(self):
        return [ObjectDetectionFineTuning(train_bn=True)]

    def _ci_benchmark_fn(self, history: List[Dict[str, Any]]) -> None:
        """
        This function is used only for debugging usage with CI
        """
        # todo (tchaton) Improve convergence
        # history[-1]["val_iou"]
