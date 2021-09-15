from typing import Callable, List

from torch import optim

from flash.core.registry import FlashRegistry

_OPTIMIZERS_REGISTRY = FlashRegistry("optimizer")

_optimizers: List[Callable] = [getattr(optim, n) for n in dir(optim) if ("_" not in n)]

for fn in _optimizers:
    _OPTIMIZERS_REGISTRY(fn, name=fn.__name__)
