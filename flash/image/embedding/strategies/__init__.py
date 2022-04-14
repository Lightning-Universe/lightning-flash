from flash.core.registry import FlashRegistry  # noqa: F401
from flash.image.embedding.strategies.default import register_default_strategy
from flash.image.embedding.strategies.vissl_strategies import register_vissl_strategies  # noqa: F401

IMAGE_EMBEDDER_STRATEGIES = FlashRegistry("embedder_training_strategies")
register_vissl_strategies(IMAGE_EMBEDDER_STRATEGIES)
register_default_strategy(IMAGE_EMBEDDER_STRATEGIES)
