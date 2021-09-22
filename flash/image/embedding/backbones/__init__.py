from flash.core.registry import FlashRegistry  # noqa: F401
from flash.image.embedding.backbones.vissl_backbones import register_vissl_backbones  # noqa: F401

IMAGE_EMBEDDER_BACKBONES = FlashRegistry("embedder_backbones")
register_vissl_backbones(IMAGE_EMBEDDER_BACKBONES)
