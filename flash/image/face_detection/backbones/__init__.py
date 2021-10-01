from flash.core.registry import FlashRegistry  # noqa: F401
from flash.image.face_detection.backbones.fastface_backbones import register_ff_backbones  # noqa: F401

FACE_DETECTION_BACKBONES = FlashRegistry("face_detection_backbones")
register_ff_backbones(FACE_DETECTION_BACKBONES)
