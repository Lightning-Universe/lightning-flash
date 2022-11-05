from flash.core.registry import FlashRegistry
from flash.text.classification.backbones.clip import CLIP_BACKBONES
from flash.text.classification.backbones.huggingface import HUGGINGFACE_BACKBONES

TEXT_CLASSIFIER_BACKBONES = FlashRegistry("backbones") + CLIP_BACKBONES + HUGGINGFACE_BACKBONES
