from flash.core.registry import FlashRegistry

from flash.image.backbones.timm import register_timm_backbones
from flash.image.backbones.resnet import register_resnet_backbones
from flash.image.backbones.transformers import register_dino_backbones
from flash.image.backbones.torchvision import (
    register_mobilenet_vgg_backbones,
    register_detection_backbones,
    register_densenet_backbones,
)


IMAGE_CLASSIFIER_BACKBONES = FlashRegistry("backbones")
OBJ_DETECTION_BACKBONES = FlashRegistry("backbones")

register_detection_backbones(OBJ_DETECTION_BACKBONES)
register_resnet_backbones(IMAGE_CLASSIFIER_BACKBONES)
register_mobilenet_vgg_backbones(IMAGE_CLASSIFIER_BACKBONES)
register_densenet_backbones(IMAGE_CLASSIFIER_BACKBONES)
register_dino_backbones(IMAGE_CLASSIFIER_BACKBONES)
register_timm_backbones(IMAGE_CLASSIFIER_BACKBONES)
