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
from functools import partial

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _ICEVISION_AVAILABLE

if _ICEVISION_AVAILABLE:
    import icevision

OBJECT_DETECTION_MODELS = FlashRegistry("backbones")
OBJECT_DETECTION_BACKBONES = FlashRegistry("backbones")

if _ICEVISION_AVAILABLE:
    # https://airctic.com/0.8.0/models/

    _yolov5 = icevision.models.ultralytics.yolov5
    _retinanet = icevision.models.torchvision.retinanet
    _faster_rcnn = icevision.models.torchvision.faster_rcnn

    def load_backbone(model_type, backbone_name: str, pretrained: bool = False):
        backbone_conf = model_type.backbones.__dict__.get(backbone_name, None)
        if backbone_conf is None:
            raise UserWarning(f"{backbone_name} is not available with {model_type}!")
        return model_type, backbone_conf(pretrained=pretrained)

    OBJECT_DETECTION_BACKBONES(partial(load_backbone, _retinanet), name='retinanet')
    OBJECT_DETECTION_BACKBONES(partial(load_backbone, _faster_rcnn), name='fasterrcnn')
    OBJECT_DETECTION_BACKBONES(partial(load_backbone, _yolov5), name='yolov5')

    @OBJECT_DETECTION_MODELS(name='icevision')
    def load_icevision(model_type, backbone_conf: icevision.backbones.BackboneConfig, num_classes: int = 1, **kwargs):
        model = model_type.model(backbone=backbone_conf, num_classes=num_classes, **kwargs)
        return model
