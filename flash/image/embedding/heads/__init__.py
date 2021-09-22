from flash.core.registry import FlashRegistry  # noqa: F401
from flash.image.embedding.heads.vissl_heads import register_vissl_heads  # noqa: F401

IMAGE_EMBEDDER_HEADS = FlashRegistry("embedder_heads")
register_vissl_heads(IMAGE_EMBEDDER_HEADS)
