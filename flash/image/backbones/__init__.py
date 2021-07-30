from flash.core.registry import FlashRegistry

from flash.core.utilities.imports import _TORCHVISION_AVAILABLE

from flash.image.backbones.resnet import register_resnet_backbones
# from flash.image.backbones.timm import register_timm_backbones
from flash.image.backbones.torchvision import (
    register_densenet_backbones,
    register_detection_backbones,
    register_mobilenet_vgg_backbones,
    register_resnext_model,
)
from flash.image.backbones.transformers import register_dino_backbones


IMAGE_CLASSIFIER_BACKBONES = FlashRegistry("backbones")
OBJ_DETECTION_BACKBONES = FlashRegistry("backbones")

register_resnet_backbones(IMAGE_CLASSIFIER_BACKBONES)
register_dino_backbones(IMAGE_CLASSIFIER_BACKBONES)

if _TORCHVISION_AVAILABLE:
    register_detection_backbones(OBJ_DETECTION_BACKBONES)
    register_mobilenet_vgg_backbones(IMAGE_CLASSIFIER_BACKBONES)
    register_resnext_model(IMAGE_CLASSIFIER_BACKBONES)
    register_densenet_backbones(IMAGE_CLASSIFIER_BACKBONES)

# if _TIMM_AVAILABLE:
#     register_timm_backbones(IMAGE_CLASSIFIER_BACKBONES)
