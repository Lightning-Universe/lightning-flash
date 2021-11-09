from flash.core.registry import FlashRegistry  # noqa: F401
from flash.image.embedding.transforms.vissl_transforms import register_vissl_transforms  # noqa: F401

IMAGE_EMBEDDER_TRANSFORMS = FlashRegistry("embedder_transforms")
register_vissl_transforms(IMAGE_EMBEDDER_TRANSFORMS)
