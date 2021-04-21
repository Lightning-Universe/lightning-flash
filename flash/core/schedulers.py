from flash.core.registry import FlashRegistry
from flash.utils.imports import _TRANSFORMERS_AVAILABLE

_SCHEDULER_REGISTRY = FlashRegistry("scheduler")

if _TRANSFORMERS_AVAILABLE:
    from transformers import optimization
    functions = [getattr(optimization, n) for n in dir(optimization) if ("get_" in n and n != 'get_scheduler')]
    for fn in functions:
        _SCHEDULER_REGISTRY(fn, name=fn.__name__[4:])
