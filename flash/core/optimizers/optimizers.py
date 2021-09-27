from inspect import isclass
from typing import Callable, List

from torch import optim

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _TORCH_OPTIMIZER_AVAILABLE

_OPTIMIZERS_REGISTRY = FlashRegistry("optimizer")

_optimizers: List[Callable] = [getattr(optim, n) for n in dir(optim) if ("_" not in n)]

for fn in _optimizers:
    _OPTIMIZERS_REGISTRY(fn, name=fn.__name__.lower())

if _TORCH_OPTIMIZER_AVAILABLE:
    import torch_optimizer

    _optimizers: List[Callable] = [
        getattr(torch_optimizer, n)
        for n in dir(torch_optimizer)
        if ("_" not in n) and isclass(getattr(torch_optimizer, n))
    ]

    for fn in _optimizers:
        name = fn.__name__.lower()
        if name not in _OPTIMIZERS_REGISTRY:
            _OPTIMIZERS_REGISTRY(fn, name=name)
