from typing import Callable, List

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _TRANSFORMERS_AVAILABLE

_SCHEDULERS_REGISTRY = FlashRegistry("scheduler")

if _TRANSFORMERS_AVAILABLE:
    from transformers import optimization

    functions: List[Callable] = [
        getattr(optimization, n) for n in dir(optimization) if ("get_" in n and n != "get_scheduler")
    ]
    for fn in functions:
        _SCHEDULERS_REGISTRY(fn, name=fn.__name__[4:])
