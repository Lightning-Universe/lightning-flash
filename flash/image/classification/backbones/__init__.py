from flash.core.registry import FlashRegistry  # noqa: F401

from flash.core.utilities.imports import _TIMM_AVAILABLE, _TORCHVISION_AVAILABLE  # noqa: F401
from flash.image.classification.backbones.resnet import register_resnet_backbones  # noqa: F401
from flash.image.classification.backbones.timm import register_timm_backbones  # noqa: F401
from flash.image.classification.backbones.torchvision import (
    register_densenet_backbones,
    register_mobilenet_vgg_backbones,
    register_resnext_model,
)  # noqa: F401
from flash.image.classification.backbones.transformers import register_dino_backbones  # noqa: F401

IMAGE_CLASSIFIER_BACKBONES = FlashRegistry("backbones")

register_resnet_backbones(IMAGE_CLASSIFIER_BACKBONES)
register_dino_backbones(IMAGE_CLASSIFIER_BACKBONES)

if _TORCHVISION_AVAILABLE:
    register_mobilenet_vgg_backbones(IMAGE_CLASSIFIER_BACKBONES)
    register_resnext_model(IMAGE_CLASSIFIER_BACKBONES)
    register_densenet_backbones(IMAGE_CLASSIFIER_BACKBONES)

if _TIMM_AVAILABLE:
    register_timm_backbones(IMAGE_CLASSIFIER_BACKBONES)
