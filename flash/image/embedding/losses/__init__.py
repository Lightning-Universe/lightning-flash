from flash.core.registry import FlashRegistry  # noqa: F401
from flash.image.embedding.losses.vissl_losses import register_vissl_losses  # noqa: F401

IMAGE_EMBEDDER_LOSS_FUNCTIONS = FlashRegistry("embedder_losses")
register_vissl_losses(IMAGE_EMBEDDER_LOSS_FUNCTIONS)
