from flash.core.registry import FlashRegistry
from flash.image.classification.backbones.clip import register_clip_backbones  # noqa: F401
from flash.image.classification.backbones.resnet import register_resnet_backbones  # noqa: F401
from flash.image.classification.backbones.timm import register_timm_backbones  # noqa: F401
from flash.image.classification.backbones.torchvision import (  # noqa: F401
    register_densenet_backbones,
    register_mobilenet_vgg_backbones,
    register_resnext_model,
)
from flash.image.classification.backbones.transformers import register_dino_backbones  # noqa: F401

IMAGE_CLASSIFIER_BACKBONES = FlashRegistry("backbones")

register_resnet_backbones(IMAGE_CLASSIFIER_BACKBONES)
register_dino_backbones(IMAGE_CLASSIFIER_BACKBONES)
register_clip_backbones(IMAGE_CLASSIFIER_BACKBONES)

register_mobilenet_vgg_backbones(IMAGE_CLASSIFIER_BACKBONES)
register_resnext_model(IMAGE_CLASSIFIER_BACKBONES)
register_densenet_backbones(IMAGE_CLASSIFIER_BACKBONES)

register_timm_backbones(IMAGE_CLASSIFIER_BACKBONES)
