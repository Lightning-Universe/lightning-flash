from flash.core.registry import FlashRegistry
from flash.utils.imports import _TRANSFORMERS_AVAILABLE

_SCHEDULER_REGISTRY = FlashRegistry("scheduler")

if _TRANSFORMERS_AVAILABLE:
    from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION

    for v in TYPE_TO_SCHEDULER_FUNCTION.values():
        _SCHEDULER_REGISTRY(v, name=v.__name__[4:])
